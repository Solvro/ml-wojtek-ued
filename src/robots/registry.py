from robots.barkour import BarkourTerrain
from robots.go1 import Go1Terrain
from robots.go2 import Go2Terrain
from robots.spot import SpotTerrain
from robots.terrain import TerrainRobot

# keyed by `Robot` value; Wojtek is not a Playground robot
TERRAIN_ROBOTS: dict[str, type[TerrainRobot]] = {
    "go1": Go1Terrain,
    "barkour": BarkourTerrain,
    "spot": SpotTerrain,
    "go2": Go2Terrain,
}
