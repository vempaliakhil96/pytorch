# mypy: allow-untyped-defs
import argparse
import sys

sizeof = {"float": 4, "at::Half": 2, "at::BFloat16": 2, "uint8_t": 1}

def unroll(uf, IndexType, InType, OutType, use_weights, isa, fused, use_offsets):
    def compute(regid, InType, use_weights, isa, prefetch):
        code = []

        if InType == "float":
            code.append(
                f"        vsum{regid} =\n"
                "            svmad_f32_x("
                f"svAll, vwgt, svld1_f32(svAll, &ip[{regid} * vLen]),"
                f" vsum{regid});"
            )
        elif InType == "at::Half":
            code.append(
                f"        vsum{regid} = svmad_f32_x(\n"
                "            svAll,\n"
                "            vwgt,\n"
                "            svcvt_f32_f16_x(\n"
                "                svAll,\n"
                "                svreinterpret_f16_u32(svld1uh_u32(\n"
                "                    svAll, reinterpret_cast<const uint16_t*>("
                f"&ip[{regid} * vLen])))),\n"  # noqa
                f"            vsum{regid});"
            )
        elif InType == "at::BFloat16":
            code.append(
                f"        vsum{regid} = svmad_f32_x(\n"
                "            svAll,\n"
                "            vwgt,\n"
                "            svreinterpret_f32_u32(svlsl_n_u32_x(\n"
                "                svAll,\n"
                "                svld1uh_u32(\n"
                "                    svAll, reinterpret_cast<const uint16_t*>("
                f"&ip[{regid} * vLen])),\n"
                "                16)),\n"  # noqa
                f"            vsum{regid});"
            )
        elif InType == "uint8_t":
            code.append(
                f"        vsum{regid} = svmad_f32_x(\n"
                "            svAll,\n"
                "            vwgt,\n"
                "            svcvt_f32_u32_x(svAll,"
                f" svld1ub_u32(svAll, &ip[{regid} * vLen])),\n"  # noqa
                f"            svadd_f32_x(svAll, vsum{regid}, vbio));"
            )
        else:
            assert False

        return code

    code = []
    code.append("    // unrolling " + str(uf) + " times")

    code.append(
        "    for ("
        + IndexType
        + " i = 0; i < output_size; ++i) {"
    )

    code.append("      " + OutType + "* const op = &out[i * block_size];")
    if use_offsets:
        code.append(
            "      if (pos != offsets[i] - offsets[0]) {\n"
            + "        return false;\n"
            + "      }"
        )
    else:
        exit(1)
    # Initialise vector sum registers
    for i in range(0, uf):
        code.append("      svfloat32_t vsum" + str(i) + " = svdup_n_f32(0);")

    # inner loop
    code.append("""\
      int64_t start_offset = offsets[i];
      int64_t end_offset = offsets[i + 1];""")
    code.append(
        "      for ("
        + "int64_t"
        + " j = start_offset; j < end_offset; ++j) {"  # noqa
    )

    code.append("        const auto idx = indices[pos];")
    code.append(
        "        if (idx < 0 || idx >= data_size) {\n"
        + "          return false;\n"
        + "        }"
    )

    if InType == "uint8_t":
        code.append("        " + OutType + " wgt = 1.f;")
        code.append("        " + OutType + " bio{};")
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (j - start_offset) : pos];"  # noqa
        )
        code.append("        }")

        if fused:
            exit(1)
        else:
            code.append("        if (scale_bias) {")
            code.append("          bio = wgt * scale_bias[2 * idx + 1];")
            code.append("          wgt = wgt * scale_bias[2 * idx];")
            code.append("        }")

        code.append("        svfloat32_t vbio = svdup_n_f32(bio);")
    else:
        code.append("        " + OutType + " wgt = 1.f;")
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (j - start_offset) : pos];"  # noqa
        )
        code.append("        }")

    code.append("        const svfloat32_t vwgt = svdup_n_f32(wgt);")
    code.append(f"        const {InType}* const ip = &input[idx * block_size];")
    code.append("        // weight * input + out")

    for i in range(0, uf):
        prefetch = False;
        code.extend(compute(i, InType, use_weights, isa, prefetch))

    code.append("        ++pos;")
    code.append("      }")

    code.append("      // Normalisation")
    code.append("      const int64_t length = end_offset - start_offset;")
    if use_offsets:
        code.append("      if (normalize_by_lengths && length != 0) {")
        code.append("        const float len_inv = 1.0f / length;")
    else:
        exit(1)

    code.append("        const svfloat32_t vlen_inv = svdup_n_f32(len_inv);")
    for i in range(0, uf):
        code.append("        svst1_f32(svAll, &op[" + str(i) + " * vLen],"
                    + " svmul_f32_x(svAll, vsum" + str(i) + ", vlen_inv));")
    code.append("      } else {")
    # inv of length
    for i in range(0, uf):
        code.append("        svst1_f32(svAll, &op[" + str(i) + " * vLen]," + " vsum" + str(i) + ");")

    code.append("      }")
    code.append("    }")
    return code


def generic(IndexType, InType, OutType, use_weights, isa, fused, use_offsets):
    def compute(InType, use_weights, isa):
        code = []
        if InType == "float":
            code.append(
                "          svst1_f32(\n"
                "              pg,\n"
                "              &op[k],\n"
                "              svmad_f32_x(\n"
                "                  pg, vwgt, svld1_f32(pg, &ip[k]),"
                " svld1_f32(pg, &op[k])));"
            )
        elif InType == "at::Half":
            code.append(
                "          svst1_f32(\n"
                "              pg,\n"
                "              &op[k],\n"
                "              svmad_f32_x(\n"
                "                  pg,\n"
                "                  vwgt,\n"
                "                  svcvt_f32_f16_x(\n"
                "                      pg,\n"
                "                      svreinterpret_f16_u32(svld1uh_u32(\n"
                "                          pg,"
                " reinterpret_cast<const uint16_t*>(&ip[k])))),\n"
                "                  svld1_f32(pg, &op[k])));"
            )
        elif InType == "at::BFloat16":
            code.append(
                "          svst1_f32(\n"
                "              pg,\n"
                "              &op[k],\n"
                "              svmad_f32_x(\n"
                "                  pg,\n"
                "                  vwgt,\n"
                "                  svreinterpret_f32_u32(svlsl_n_u32_x(\n"
                "                      pg,\n"
                "                      svld1uh_u32(\n"
                "                          pg,"
                " reinterpret_cast<const uint16_t*>(&ip[k])),\n"
                "                      16)),\n"
                "                  svld1_f32(pg, &op[k])));"
            )
        elif InType == "uint8_t":
            code.append(
                "          svst1_f32(\n"
                "              pg,\n"
                "              &op[k],\n"
                "              svmad_f32_x(\n"
                "                  pg,\n"
                "                  vwgt,\n"
                "                  svcvt_f32_u32_x(pg,"
                " svld1ub_u32(pg, &ip[k])),\n"  # noqa
                "                  svadd_f32_x(pg,"
                " svld1_f32(pg, &op[k]), vbio)));"
            )
        else:
            assert False

        return code

    code = []

    if use_offsets:
        code.append(
            "    for ("
            + IndexType
            + " i = 0; i < output_size; ++i) {"
        )
    else:
        exit(1)

    code.append("      " + OutType + "* const op = &out[i * block_size];")

    # initialize to 0
    code.append("      memset(op, 0, sizeof(float) * block_size);")

    # inner loop
    if use_offsets:
        code.append(
            "      if (pos != offsets[i] - offsets[0]) {\n"
            + "        return false;\n"
            + "      }"
        )
        code.append("""\
      int64_t start_offset = offsets[i];
      int64_t end_offset = offsets[i + 1];""")
        code.append(
            "      for ("
            + "int64_t"
            + " j = start_offset; j < end_offset; ++j) {"  # noqa
        )
    else:
        exit(1)

    code.append("        const auto idx = indices[pos];")
    code.append(
        "        if (idx < 0 || idx >= data_size) {\n"
        + "          return false;\n"
        + "        }"
    )

    if InType == "uint8_t":
        code.append("        // unimplemented")
        code.append("        " + OutType + " wgt = 1.f;")
        code.append("        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)")
        code.append("        " + OutType + " bio{};")
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (j - start_offset) : pos];"  # noqa
        )
        code.append("        }")
        if fused:
            exit(1)
        else:
            code.append("        if (scale_bias) {")
            code.append("          bio = wgt * scale_bias[2 * idx + 1];")
            code.append("          wgt = wgt * scale_bias[2 * idx];")
            code.append("        }")
        code.append("        svfloat32_t vbio = svdup_n_f32(bio);")
    else:
        code.append("        " + OutType + " wgt = 1.f;")
        code.append("        if (weights) {")
        code.append(
            "          wgt = weights[IS_WEIGHT_POSITIONAL ? (j - start_offset) : pos];"  # noqa
        )
        code.append("        }")
    code.append("        const svfloat32_t vwgt = svdup_n_f32(wgt);")

    code.append(f"        const {InType}* ip = &input[idx * block_size];")

    # compute and store main loop
    code.append("        svbool_t pg;")
    code.append("        for (int64_t k = 0;")
    code.append("             svptest_first(svAll, pg = svwhilelt_b32_s64("
                + "k, block_size));")
    code.append("             k += vLen) {")
    code.extend(compute(InType, use_weights, isa))
    code.append("        }\n")
    code.append("        ++pos;")
    code.append("      }")

    if use_offsets:
        code.append("      const int64_t length = end_offset - start_offset;\n")
        code.append("      if (normalize_by_lengths && length != 0) {")
        code.append("        const float len_inv = 1.0f / length;")
    else:
        exit(1)

    code.append("        svfloat32_t vlen_inv = svdup_n_f32(len_inv);")
    code.append("        svbool_t pg;")
    code.append("        for (int64_t j = 0;\n"
                "             svptest_first(svAll, pg = svwhilelt_b32_s64("
                "j, block_size));")
    code.append("             j += vLen) {")
    code.append(
        "          svst1_f32(\n"
        "              pg, &op[j], svmul_f32_x(pg, svld1_f32(pg, &op[j]), vlen_inv));"
    )
    code.append("        }")
    code.append("      }")
    code.append("    }")
    return code


# start main code
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--filename", help="file name")
parser.add_argument("--fused", action="store_true")
parser.add_argument("--use-offsets", action="store_true")
opts = parser.parse_args()
if opts.filename:
    filename = opts.filename
elif opts.fused:
    print("This script currently does not support --fused")
    exit(1)
else:
    if opts.use_offsets:
        filename = "embedding_lookup_idx_sve.cc"
    else:
        print("This script currently only supports codegen with --use-offsets")
        exit(1)
        filename = "embedding_lookup_sve.cc"

options = [
    ["int32_t", "int32_t", "float", "float", "float", "float"],
    ["int64_t", "int64_t", "float", "float", "float", "float"],
    ["int32_t", "int32_t", "half", "at::Half", "float", "float"],
    ["int64_t", "int64_t", "half", "at::Half", "float", "float"],
    ["int32_t", "int32_t", "bfloat16", "at::BFloat16", "float", "float"],
    ["int64_t", "int64_t", "bfloat16", "at::BFloat16", "float", "float"],
    ["int32_t", "int32_t", "uint8_t", "uint8_t", "float", "float"],
    ["int64_t", "int64_t", "uint8_t", "uint8_t", "float", "float"],
]

code = []
# includes
code.append("//// --------------------------")
code.append("//// ATTENTION:")
code.append("//// THIS CODE IS AUTOGENERATED")
code.append(f"//// BY {sys.argv[0]}")
code.append("//// DO NOT MODIFY!!!")
code.append("//// --------------------------\n")

code.append("#include <arm_sve.h>")
code.append("#include <c10/util/BFloat16.h>")
code.append("#include <c10/util/Half.h>")
code.append("#include <cstdint>")
code.append("#include <cstring>")

code.append("namespace caffe2 {\n")
for o in options:
    [IndexTypeName, IndexType, InTypeName, InType, OutTypeName, OutType] = o

    prefix = "Fused8BitRowwise" if opts.fused else ""
    code.append("template <bool IS_WEIGHT_POSITIONAL>")
    if opts.use_offsets:
        fn_base = f"{prefix}EmbeddingLookupIdx_{IndexTypeName}_{InTypeName}_{OutTypeName}"
    else:
        exit(1)

    suffix = "__sve"
    fn = "static bool " + fn_base + suffix
    code.append(fn + "(")

    args = []
    args.append("    const int64_t block_size,")
    args.append("    const int64_t output_size,")
    args.append("    const int64_t index_size,")
    args.append("    const int64_t data_size,")
    args.append("    const " + InType + "* input,")
    args.append("    const " + IndexType + "* indices,")
    if opts.use_offsets:
        args.append("    const " + IndexType + "* offsets,")
    else:
        exit(1)
    args.append("    const float* weights,")
    if not opts.fused:
        args.append("    const float* scale_bias,")
    args.append("    bool normalize_by_lengths,")
    args.append("    " + OutType + "* out) {")
    code += args

    code.append("  const svbool_t svAll = svptrue_b32();")
    code.append("  const auto vLen = static_cast<int64_t>(svcntw());")

    code.append("  // NOLINTNEXTLINE(cppcoreguidelines-narrowing-conversions,bugprone-narrowing-conversions)")
    # block_size is the number of elements and fused_block_size is the size of
    # an entire row, including scale and bias.
    offset = (8 // sizeof[InType]) if opts.fused else 0
    #code.append( f"  const {IndexType} fused_block_size = block_size + {offset};")
    if opts.use_offsets:
        code.append("  int64_t pos = 0;")
    else:
        exit(1)

    # code.append("printf(\"calling " + fn + "\\n\");");

    code.append("  if (block_size == 32 * vLen) {")
    code += unroll(32, IndexType, InType, OutType, True, "SVE", opts.fused, opts.use_offsets)
    code.append("  } else if (block_size == 16 * vLen) {")
    code += unroll(16, IndexType, InType, OutType, True, "SVE", opts.fused, opts.use_offsets)
    code.append("  } else if (block_size == 8 * vLen) {")
    code += unroll(8, IndexType, InType, OutType, True, "SVE", opts.fused, opts.use_offsets)
    code.append("  } else if (block_size == 4 * vLen) {")
    code += unroll(4, IndexType, InType, OutType, True, "SVE", opts.fused, opts.use_offsets)
    code.append("  } else if (block_size == 2 * vLen) {")
    code += unroll(2, IndexType, InType, OutType, True, "SVE", opts.fused, opts.use_offsets)
    code.append("  } else {")
    code.append("    // generic code:")
    code.append("    // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-avoid-c-arrays)")
    code += generic(IndexType, InType, OutType, True, "SVE", opts.fused, opts.use_offsets)
    code.append("  }")
    code.append("  return pos == index_size;")

    code.append("}")

    for is_weight_positional in ["false", "true"]:
        code.append("bool " + fn_base + "_" + is_weight_positional + suffix + "(")
        code += args
        # Resolve the Lint warnings: Limit of 80 characters in one line.
        extra_space = "\n      "
        ret_string = "  return " + fn_base + suffix + "<" + is_weight_positional + ">("
        if len(ret_string) <= 80:
            code.append(ret_string)
        else:
            code.append("  return " + fn_base + suffix + "<" + extra_space + is_weight_positional + ">(")
        code.append("      block_size,")
        code.append("      output_size,")
        code.append("      index_size,")
        code.append("      data_size,")
        code.append("      input,")
        code.append("      indices,")
        if opts.use_offsets:
            code.append("      offsets,")
        else:
            exit(1)
        code.append("      weights,")
        if not opts.fused:
            code.append("      scale_bias,")
        code.append("      normalize_by_lengths,")
        code.append("      out);")
        code.append("}")

    code.append("")

code.append("} // namespace caffe2")

with open(filename, "w") as fout:
    for c in code:
        # print(c, file = fout)
        fout.write(c + "\n")


print("Created " + filename)
