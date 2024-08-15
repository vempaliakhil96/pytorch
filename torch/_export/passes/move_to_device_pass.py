from typing import Dict, Union

import torch
from torch.export import ExportedProgram


def _move_by_dict(
    v: Union[torch.nn.Parameter, torch.Tensor], location_dict: Dict[str, str]
) -> Union[torch.nn.Parameter, torch.Tensor]:
    if str(v.device) in location_dict.keys():
        return v.to(location_dict[str(v.device)])
    else:
        return v


def _move_nn_param(
    v: torch.nn.Parameter, location: Union[torch.device, str, Dict[str, str]]
) -> torch.nn.Parameter:
    if isinstance(location, dict):
        return torch.nn.Parameter(_move_by_dict(v, location))
    else:
        return torch.nn.Parameter(v.to(location))


def _move_tensor(
    v: torch.Tensor, location: Union[torch.device, str, Dict[str, str]]
) -> torch.Tensor:
    if isinstance(location, dict):
        return _move_by_dict(v, location)
    else:
        return v.to(location)


def _move_kwargs(
    kwargs: Dict[str, object], location: Union[torch.device, str, Dict[str, str]]
) -> Dict[str, object]:
    new_kwargs = kwargs.copy()
    if isinstance(location, dict):
        if str(kwargs["device"]) in location.keys():
            new_kwargs["device"] = location[str(kwargs["device"])]
    else:
        new_kwargs["device"] = location
    return new_kwargs


def move_to_device_pass(
    ep: ExportedProgram, location: Union[torch.device, str, Dict[str, str]]
) -> ExportedProgram:
    """
    Move the exported program to the given device.

    Args:
        ep (ExportedProgram): The exported program to move.
        location (Union[torch.device, str, Dict[str, str]]): The device to move the exported program to.
            If a string, it is interpreted as a device name.
            If a dict, it is interpreted as a mapping from
            the existing device to the intended one

    Returns:
        ExportedProgram: The moved exported program.
    """
    # move all the state_dict to the new location
    for k, v in ep.state_dict.items():
        if isinstance(v, torch.nn.Parameter):
            ep._state_dict[k] = _move_nn_param(v, location)
        else:
            ep._state_dict[k] = _move_tensor(v, location)

    # move all the constants
    for k, v in ep.constants.items():
        if isinstance(v, torch.Tensor):
            ep._constants[k] = _move_tensor(v, location)

    for node in ep.graph.nodes:
        # move all the nodes with burnt-in device kwargs
        if "device" in node.kwargs:
            node.kwargs = _move_kwargs(node.kwargs, location)
        # move all the tensor metadata
        for k, v in node.meta.items():
            if isinstance(v, torch.Tensor):
                v = _move_tensor(v, location)
            elif isinstance(v, tuple):
                for v_i in v:
                    if isinstance(v_i, torch.Tensor):
                        v_i = _move_tensor(v_i, location)
    ep.validate()
    return ep
