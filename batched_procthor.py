import random
from typing import Dict, Any, Tuple, Sequence
import copy
import time
import threading as th

from ai2thor.controller import Controller
import prior
from matplotlib import pyplot as plt
from procthor.utils.upgrade_house_version import HouseUpgradeManager
import numpy as np

# TODO Additional requirements:
#  pip install prior matplotlib pandas attr attrs
#  pip install -e "git+https://github.com/allenai/procthor.git@release-api#egg=procthor"

# TODO local build from
#  UNITY_BUILD_NAME=jordis_osx /Applications/Unity/Unity.app/Contents/MacOS/Unity -quit -batchmode
#  -logFile /Users/jordis/work/ai2thor-jordis/thor-OSXIntel64-local.log
#  -projectpath /Users/jordis/work/ai2thor-jordis/unity -buildTarget OSXUniversal -executeMethod Build.OSXIntel64

# TODO GetReachablePositions only takes into account agent0's reachable positions.
#  Using a single agent requires teleporting before any controller step - overkill. Abandon?

# Ideas from batch rendering. Assume agent s will observe o_{t-1} when deciding for a_{t+1}, so that we can use
# third party cameras to render o_t, while we use a full agent to perform the navigation (with rendering disabled).
# We'll have two threads: one rendering the output pose from sim with a camera,
# the other one simulating the action. With processes, the memory usage increases by 2, and so the time to load assets (unless we exclude textures fro simulator).

LOCAL_THOR_PATH = "/Users/jordis/work/ai2thor-jordis/unity/jordis_osx.app/Contents/MacOS/AI2-THOR"

SPACING = 50.0
NAME_MOD_PREFIX = ".rep"


def rename_wall(wall, shiftx: float, shiftz: float, name_mod: str) -> str:
    wall_parts = wall.split("|")
    name = "|".join(wall_parts[:2]) + name_mod  # parts 0, 1
    x0 = f"{float(wall_parts[2]) + shiftx:.2f}"
    z0 = f"{float(wall_parts[3]) + shiftz:.2f}"
    x1 = f"{float(wall_parts[4]) + shiftx:.2f}"
    z1 = f"{float(wall_parts[5]) + shiftz:.2f}"
    return "|".join([name, x0, z0, x1, z1])


def id_to_shifts(house_id) -> Tuple[float, float]:
    shiftx = SPACING * (house_id // 4)
    shiftz = SPACING * (house_id % 4)
    return shiftx, shiftz


def id_to_name_mod(house_id: int) -> str:
    return NAME_MOD_PREFIX + f"{house_id}"


def add_replica(
    houses: Dict[str, Any],
    house: Dict[str, Any],
    house_id: int,
    height: float = -1.0,
) -> Dict[str, Any]:
    assert height > 0

    shiftx, shiftz = id_to_shifts(house_id)
    name_mod = id_to_name_mod(house_id)

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
        if "cullingMaskOff" in new_light:
            new_light["cullingMaskOff"] = []
        new_lights.append(new_light)

    new_rooms = []
    for it, room in enumerate(house["rooms"]):
        new_room = copy.deepcopy(room)
        new_room["id"] += name_mod
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


def shift_agent(agent_meta: Dict[str, Any], house_id: int) -> Dict[str, Any]:
    shiftx, shiftz = id_to_shifts(house_id)

    shifted_meta = {**agent_meta}
    shifted_meta["position"] = {**shifted_meta["position"]}
    shifted_meta["position"]["x"] += shiftx
    shifted_meta["position"]["z"] += shiftz
    return shifted_meta


class BatchController:
    DEFAULT_FOVS = dict(default=90.0, locobot=60.0)

    def __init__(self, **thor_kwargs):
        self.thor_kwargs = {**thor_kwargs, "scene": "Procedural"}

        self.agent_mode = self.thor_kwargs.get("agentMode", "default")
        self.field_of_view = self.thor_kwargs.get("fieldOfView", self.DEFAULT_FOVS[self.agent_mode])

        self.controller = Controller(**self.thor_kwargs)
        self.single_houses = []
        self.agent_poses = []
        self.events = []
        self.all_reset_object_ids = []
        self.lock = th.Lock()

    def reset(self, **kwargs):
        self.controller.reset(**{**kwargs, "scene": "Procedural"})

        self.single_houses = []
        self.agent_poses = []
        self.events = []
        self.all_reset_object_ids = []

    def create_houses(self, *args, **kwargs):
        single_houses = kwargs["house"] if "house" in kwargs else args[1]

        self.single_houses = single_houses
        self.agent_poses = [
            shift_agent(single_houses[hit]["metadata"]["agent"], house_id=hit)
            for hit in range(len(single_houses))
        ]
        self.all_reset_object_ids = [[] for _ in range(len(self.single_houses))]

        # self.controller.step(
        #     action="AddThirdPartyCamera",
        #     position=self.agent_poses[0]["position"],
        #     rotation=self.agent_poses[0]["rotation"],  # ignore horizon
        #     fieldOfView=self.field_of_view,
        #     renderImage=False,
        # )

        height = max([h['walls'][0]['polygon'][2]['y'] for h in single_houses])

        merged_houses = dict(metadata=None)

        # Use one of the boxes and keep the directional light that was used with the chosen box
        proc_hit = random.randint(0, len(single_houses) - 1)
        merged_houses["proceduralParameters"] = {**single_houses[proc_hit]["proceduralParameters"]}
        merged_houses["proceduralParameters"]["lights"] = [
            {**single_houses[proc_hit]["proceduralParameters"]["lights"][0]}
        ]

        # Then add shifted replicas of all current houses
        for hit in range(len(single_houses)):
            merged_houses = add_replica(merged_houses, single_houses[hit], house_id=hit, height=height)

        house_to_load = HouseUpgradeManager.upgrade_to(merged_houses, "1.0.0")
        self.controller.step(action="CreateHouse", house=house_to_load, renderImage=False, raise_for_failure=True)

        all_reset_object_ids = [o["objectId"] for o in self.controller.last_event.metadata["objects"]]
        for oid in all_reset_object_ids:
            if oid == "Floor" or oid.startswith("wall|"):
                continue
            actual_oid, hit = oid.split(NAME_MOD_PREFIX)
            hit = hit.split("___")[0]
            self.all_reset_object_ids[int(hit)].append(oid)

        self.controller.step(action="SetObjectFilter", objectIds=[], renderImage=False, raise_for_failure=True)

    @staticmethod
    def _adapt_state(state):
        return dict(
            position=state["position"],
            rotation=state["rotation"],
            horizon=state["cameraHorizon"],
            standing=state["isStanding"],
        )

    def simulate(self, actions):
        for hit in range(len(self.single_houses)):
            if "Teleport" not in actions[hit]:
                self.lock.acquire()
                self.controller.step(
                    action="TeleportFull",
                    **self.agent_poses[hit],
                    renderImage=False,
                    forceAction=True,
                )
                self.lock.release()
            self.lock.acquire()
            evt = self.controller.step(actions[hit], renderImage=False)
            self.lock.release()
            self.agent_poses[hit] = self._adapt_state(evt.metadata["agent"])

    def simulate_and_render(self, actions):
        self.events = []
        for hit in range(len(self.single_houses)):
            if "Teleport" not in actions[hit]:
                self.controller.step(
                    action="TeleportFull",
                    **self.agent_poses[hit],
                    renderImage=False,
                    forceAction=True,
                )
            evt = self.controller.step(actions[hit])
            self.agent_poses[hit] = self._adapt_state(evt.metadata["agent"])
            self.events.append(evt)

    def render(self, last_states):
        self.events = []
        for hit in range(len(self.single_houses)):
            self.lock.acquire()
            evt = self.controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=last_states[hit]["position"],
                rotation=last_states[hit]["rotation"],  # TODO missing horizon
                fieldOfView=self.field_of_view,
            )
            self.lock.release()
            self.events.append(evt)

    def step(self, *args, **kwargs):
        if args[0] == "CreateHouse" or "action" in kwargs and kwargs["action"] == "CreateHouse":
            self.create_houses(*args, **kwargs)
            self.controller.step(
                action="TeleportFull",
                **self.agent_poses[-1],
                renderImage=False,
                forceAction=True,
            )
            return None

        last_states = self.agent_poses[:]
        actions = kwargs["action"] if "action" in kwargs else args[0]

        # sim = th.Thread(target=self.simulate, args=(actions,))
        # sim.start()
        # self.render(last_states)
        # sim.join()

        self.simulate_and_render(actions)

        return self.events

    def stop(self):
        self.controller.stop()


def test_multi_houses():
    data = prior.load_dataset("procthor-10k")

    bc = BatchController(rotateStepDegrees=30, snapToGrid=False)

    single_houses = [copy.deepcopy(data["train"][it]) for it in range(6)]

    fig, ax = plt.subplots(4, 4)

    stime = time.time()
    bc.step("CreateHouse", house=single_houses)
    print(f"create houses {time.time() - stime:.2f} s")

    times = []
    for step in range(20):
        stime = time.time()
        actions = random.choices(["MoveAhead", "RotateLeft", "RotateRight"], k=len(single_houses))
        evts = bc.step(actions)
        times.append(time.time()-stime)
        print(f"step {step} actions {actions} in {times[-1]:.2f} s")
        for hit in range(len(evts)):
            # ax[hit // 4, hit % 4].imshow(evts[hit].third_party_camera_frames[0])
            ax[hit // 4, hit % 4].imshow(evts[hit].frame)
        # print()

    print(f"mean time {np.mean(times[1:]):.2f} s")

    bc.stop()
    print("DONE batched")

    single_houses = [copy.deepcopy(data["train"][it]) for it in range(len(single_houses))]

    cs = [
        Controller(rotateStepDegrees=30, snapToGrid=False, scene="Procedural")
        for _ in range(len(single_houses))
    ]

    fig, ax = plt.subplots(4, 4)

    stime = time.time()
    for c, single_house in zip(cs, single_houses):
        c.step("CreateHouse", house=HouseUpgradeManager.upgrade_to(single_house, "1.0.0"))
        c.step(
            action="TeleportFull",
            **single_house["metadata"]["agent"],
            renderImage=False,
            forceAction=True,
        )
        c.step(action="SetObjectFilter", objectIds=[], raise_for_failure=True)
    print(f"create houses {time.time() - stime:.2f} s")

    times = []
    for step in range(20):
        stime = time.time()
        actions = random.choices(["MoveAhead", "RotateLeft", "RotateRight"], k=len(single_houses))
        evts = []
        for c, action in zip(cs, actions):
            # c.step(
            #     action="TeleportFull",
            #     **BatchController._adapt_state(c.last_event.metadata["agent"]),
            #     renderImage=False,
            #     forceAction=True,
            # )
            evts.append(c.step(action))
        times.append(time.time()-stime)
        print(f"step {step} actions {actions} in {times[-1]:.2f} s")
        for hit in range(len(evts)):
            ax[hit // 4, hit % 4].imshow(evts[hit].frame)

    print(f"mean time {np.mean(times[1:]):.2f} s")

    for c in cs:
        c.stop()
    print("DONE iterative")


if __name__ == "__main__":
    test_multi_houses()
