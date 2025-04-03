import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image
import pdb
import tqdm

'''
This script animates the paths of agents in the map.
'''

def create_gif(image_folder, output_path, duration=100, end_frame_duration=2000):
    images: list[Image.Image] = []
    for file_name in sorted(os.listdir(image_folder)):
        if file_name.endswith(('png')):
            file_path = os.path.join(image_folder, file_name)
            images.append(Image.open(file_path))

    if images:
        duration = [duration] * (len(images) - 1) + [end_frame_duration] # Make last frame longer
        images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    else:
        print("No images found in the folder")
    
    image_names = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        os.remove(image_path)


def parse_scene(scen_file):
    """Input: scenfile
    Output: start_locations, goal_locations
    """
    start_locations = []
    goal_locations = []

    with open(scen_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('version'):
                continue
            tokens = line.split("\t")
            # pdb.set_trace()
            assert(len(tokens) == 9)
            tokens = tokens[4:]
            col = int(tokens[0])
            row = int(tokens[1])
            start_locations.append((row, col))
            col = int(tokens[2])
            row = int(tokens[3])
            goal_locations.append((row, col))
    return np.array(start_locations, dtype=int), np.array(goal_locations, dtype=int)

def readMap(mapfile: str):
    """ Read map """
    if mapfile.startswith("../data"):
        mapfile = mapfile[3:]
    with open(mapfile) as f:
        line = f.readline()
        line = f.readline()
        height = int(line.split(' ')[1])
        line = f.readline()
        width = int(line.split(' ')[1])
        line = f.readline()
        mapdata = np.array([list(line.rstrip()) for line in f])

    mapdata.reshape((width, height))
    mapdata[mapdata == '.'] = 0
    mapdata[mapdata == '@'] = 1
    mapdata[mapdata == 'T'] = 1
    mapdata = mapdata.astype(int)
    return mapdata


def createAnimation(args):
    mapName = args.mapName
    mapdata = readMap(f"{args.mapFolder}/{mapName}.map")
    id2plan = np.load(args.pathsNpyFilePath) # (T,N,2)
    max_plan_length = id2plan.shape[0]
    num_agents = id2plan.shape[1]
    # pdb.set_trace()
    if args.scenName is not None:
        scen_abbr = args.scenName.split(".")[0]
        scen_file = f"{args.sceneFile}/{scen_abbr}.scen"
        start_locs, id2goal = parse_scene(scen_file)
        success = np.all(id2plan[-1] == id2goal[0:num_agents])
        if success:
            textColor = 'green'
        else:
            textColor = 'red'
    else:
        success = None
        textColor = 'black'
    outputFilePath = args.outputGif
    
    tmpFolder = args.tmpFolderToSaveImages
    os.makedirs(tmpFolder, exist_ok=True)
    
    colors = ['r', 'b', 'm', 'g']
    last_row = id2plan[-1]
    repeated_rows = np.tile(last_row, (40, 1, 1))
    id2plan = np.vstack([id2plan, repeated_rows])
    for t in tqdm.tqdm(range(0, max_plan_length), desc="Creating visualization"):
        plt.imshow(mapdata, cmap="Greys")
        plt.xticks([])
        plt.yticks([])
        for i in range(num_agents):
            plan = id2plan[:, i]
            if success is not None and np.all(plan[t] == id2goal[i]):
                plt.scatter(plan[t][1], plan[t][0], s=3, c="grey")
            else:
                plt.scatter(plan[t][1], plan[t][0], s=3, c=colors[i % len(colors)])
        plt.subplots_adjust(top=0.85)
        name = "{}/{:03d}.png".format(tmpFolder, t)
        plt.title(f"{mapName}: t = {t}", color=textColor)
        plt.savefig(name)
        plt.cla()
    
    create_gif(tmpFolder, outputFilePath)


"""
Example usage:
python -m main_pys.visualize_path den312d logs/paths.npy --scenName=den312d-random-1.scen 
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize agent paths from log file')
    parser.add_argument("mapName", help="map name without .map, needs to be in mapsToMaxNumAgents defined in the top", type=str) # Note: Positional is required
    parser.add_argument("pathsNpyFilePath", help="Path to the paths.npy file", type=str) # Note: Positional is required
    parser.add_argument("--scenName", help="scen name with .scen", type=str, default=None) # Note: Positional is required
    parser.add_argument("--mapFolder", help="folder containing maps", type=str, default="data/mapf-map")
    parser.add_argument("--sceneFile", help="folder containing maps", type=str, default="data/mapf-scen-random")
    parser.add_argument('--tmpFolderToSaveImages', type=str, help='temporary folder to save images', default="data/tmpFolder")
    parser.add_argument('--outputGif', type=str, help='Path to the output gif file', default="logs/paths.gif")
    args = parser.parse_args()
    createAnimation(args)
