import random
from typing import Dict, Any
import copy
import time

from ai2thor.controller import Controller
import prior
from matplotlib import pyplot as plt
from procthor.utils.upgrade_house_version import HouseUpgradeManager

# TODO Additional requirements:
#  pip install prior matplotlib pandas attr attrs
#  pip install -e "git+https://github.com/allenai/procthor.git@release-api#egg=procthor"

SPACING = 50.0
LOCAL_THOR_PATH = "/Users/jordis/work/ai2thor-jordis/unity/jordis_osx.app/Contents/MacOS/AI2-THOR"


def rename_wall(wall, shiftx: float, shiftz: float, name_mod: str):
    wall_parts = wall.split("|")
    name = "|".join(wall_parts[:2]) + name_mod  # parts 0, 1
    x0 = f"{float(wall_parts[2]) + shiftx:.2f}"
    x1 = f"{float(wall_parts[4]) + shiftx:.2f}"
    z0 = f"{float(wall_parts[3]) + shiftz:.2f}"
    z1 = f"{float(wall_parts[5]) + shiftz:.2f}"
    return "|".join([name, x0, z0, x1, z1])


def add_replica(
    houses: Dict[str, Any],
    house: Dict[str, Any],
    house_id: int,
    shift: float = SPACING,
    name_mod: str = ".rep",
    height: float = -1.0,
):
    assert height > 0

    # shift = shift * house_id
    shiftx = shift * (house_id // 4)
    shiftz = shift * (house_id % 4)
    del shift
    name_mod = name_mod + f"{house_id}"

    new_doors = []
    for door in house["doors"]:
        new_door = copy.deepcopy(door)
        new_door["id"] += name_mod
        new_door["room0"] += name_mod
        new_door["room1"] += name_mod
        new_door["wall0"] = rename_wall(new_door["wall0"], shiftx, shiftz, name_mod)
        new_door["wall1"] = rename_wall(new_door["wall1"], shiftx, shiftz, name_mod)
        new_doors.append(new_door)

    new_objects = [copy.deepcopy(obj) for obj in house["objects"]]
    to_rep = new_objects[:]
    while len(to_rep):
        new_to_rep = []
        for new_obj in to_rep:
            new_obj["id"] += name_mod
            new_obj["position"]["x"] += shiftx
            new_obj["position"]["z"] += shiftz
            if "layer" in new_obj:
                new_obj["layer"] += name_mod
            else:
                # print(f"No layer for obj in {name_mod}")
                new_obj["layer"] = "global_layer" + name_mod
            if "children" in new_obj:
                new_to_rep.extend(new_obj["children"])
        to_rep = new_to_rep

    new_lights = []
    for light in house["proceduralParameters"]["lights"]:
        if light["type"] == "directional":
            continue
        if light["type"] != "point":
            print(f"Unknown light type {light['type']}")
            continue
        new_light = copy.deepcopy(light)
        new_light["id"] += name_mod
        new_light["position"]["x"] += shiftx
        new_light["position"]["z"] += shiftz
        if "layer" in new_light:
            new_light["layer"] += name_mod
        else:
            # print(f"No layer for light in {name_mod}")
            new_light["layer"] = "global_layer" + name_mod
        # if "cullingMaskOff" in new_light:
        #     for it in range(len(new_light["cullingMaskOff"])):
        #         new_light["cullingMaskOff"][it] = new_light["cullingMaskOff"][it] + name_mod
        if "cullingMaskOff" in new_light:
            new_light["cullingMaskOff"] = []
        new_lights.append(new_light)

    new_rooms = []
    for it, room in enumerate(house["rooms"]):
        new_room = copy.deepcopy(room)
        new_room["id"] += name_mod
        if "layer" in new_room:
            new_room["layer"] += name_mod
        else:
            # print(f"No layer for room in {name_mod}")
            new_room["layer"] = "global_layer" + name_mod
        for vertex in new_room["floorPolygon"]:
            vertex["x"] += shiftx
            vertex["z"] += shiftz
        new_rooms.append(new_room)

    new_walls = []
    for wall in house["walls"]:
        new_wall = copy.deepcopy(wall)
        new_wall["id"] = rename_wall(new_wall["id"], shiftx, shiftz, name_mod)
        for vertex in new_wall["polygon"]:
            vertex["x"] += shiftx
            vertex["z"] += shiftz
            if vertex["y"] > 0.2:
                vertex["y"] = height
        new_wall["roomId"] += name_mod
        if "layer" in new_wall:
            new_wall["layer"] += name_mod
        else:
            # print(f"No layer for wall in {name_mod}")
            new_wall["layer"] = "global_layer" + name_mod
        new_walls.append(new_wall)

    new_windows = []
    for window in house["windows"]:
        new_window = copy.deepcopy(window)
        new_window["id"] += name_mod
        new_window["room0"] += name_mod
        new_window["room1"] += name_mod
        new_window["wall0"] = rename_wall(new_window["wall0"], shiftx, shiftz, name_mod)
        new_window["wall1"] = rename_wall(new_window["wall1"], shiftx, shiftz, name_mod)
        new_windows.append(new_window)

    if "doors" not in houses:
        houses["doors"] = []
    houses["doors"].extend(new_doors)

    if "objects" not in houses:
        houses["objects"] = []
    houses["objects"].extend(new_objects)

    if "lights" not in houses["proceduralParameters"]:
        houses["proceduralParameters"]["lights"] = []
    houses["proceduralParameters"]["lights"].extend(new_lights)

    if "rooms" not in houses:
        houses["rooms"] = []
    houses["rooms"].extend(new_rooms)

    if "walls" not in houses:
        houses["walls"] = []
    houses["walls"].extend(new_walls)

    if "windows" not in houses:
        houses["windows"] = []
    houses["windows"].extend(new_windows)

    return houses


def shift_agent(agent_meta, house_id: int, shift: float = SPACING):
    shiftx = shift * (house_id // 4)
    shiftz = shift * (house_id % 4)
    del shift

    shifted_meta = {**agent_meta}
    shifted_meta["position"] = {**shifted_meta["position"]}
    shifted_meta["position"]["x"] += shiftx
    shifted_meta["position"]["z"] += shiftz
    return shifted_meta


def culling_masks(houses):
    all_layers = set()
    for light in houses["proceduralParameters"]["lights"]:
        if "layer" in light:
            all_layers.add(light["layer"])
        # else:
        #     print(f"light with no layer: {light}")

    print(f"Using {len(all_layers)} layers")

    # TODO unity/Assets/Scripts/BaseFPSAgentController.cs (probably others as well) uses hard-coded mask names!

    scene_to_layers = {}
    for layer in all_layers:
        scene = layer.split(".")[-1]
        if scene not in scene_to_layers:
            scene_to_layers[scene] = []
        scene_to_layers[scene].append(layer)

    # print(f"scene_to_layers {scene_to_layers}")

    for light in houses["proceduralParameters"]["lights"][1:]:
        # If a light has `global_layer`, we should leave all layers in current scene!
        if "cullingMaskOff" in light and "global_layer" not in light["layer"]:
            light["cullingMaskOff"] = list(all_layers - {light["layer"]})
        else:
            scene = light["layer"].split(".")[-1]
            light["cullingMaskOff"] = list(all_layers - set(scene_to_layers[scene]))

    return houses


def test_multi_houses():
    data = prior.load_dataset("procthor-10k")

    single_houses = [data["train"][it] for it in range(7)]
    # single_houses = [data["train"][it] for it in [8]]

    stime = time.time()

    height = max([h['walls'][0]['polygon'][2]['y'] for h in single_houses])

    houses = dict(metadata=None)
    proc_hit = random.randint(0, len(single_houses) - 1)
    houses["proceduralParameters"] = {**single_houses[proc_hit]["proceduralParameters"]}
    # set the directional light formerly used with the current box
    houses["proceduralParameters"]["lights"] = [
        {**single_houses[proc_hit]["proceduralParameters"]["lights"][0]}
    ]

    for hit in range(len(single_houses)):
        houses = add_replica(houses, single_houses[hit], house_id=hit, height=height)
    houses = culling_masks(houses)

    # TODO - we'll need to spatially filter object Ids

    house_to_load = HouseUpgradeManager.upgrade_to(houses, "1.0.0")

    print(f"Merged in {time.time() - stime:.4f} s")

    c = Controller(
        local_executable_path=LOCAL_THOR_PATH,
        scene="Procedural",
        agentCount=len(single_houses),
    )
    stime = time.time()
    c.step(action="CreateHouse", house=house_to_load, raise_for_failure=True)
    print(f"Created in {time.time()-stime:.4f} s")
    stime = time.time()
    for hit in range(len(single_houses)):
        c.step(
            action="TeleportFull",
            **shift_agent(single_houses[hit]["metadata"]["agent"], house_id=hit),
            agentId=hit,
            renderImage=hit == len(single_houses) - 1
        )
    print(f"Teleported in {time.time() - stime:.4f} s")

    fig, ax = plt.subplots(4, 4)
    for hit in range(len(single_houses)):
        ax[hit // 4, hit % 4].imshow(c.last_event.events[hit].frame)
    # plt.show()

    event = c.step(action="GetMapViewCameraProperties")

    c.step(action="AddThirdPartyCamera", **event.metadata["actionReturn"])
    plt.imshow(c.last_event.events[0].third_party_camera_frames[0])

    print("DONE")


if __name__ == "__main__":
    test_multi_houses()
