import os
import argparse
import subprocess  # For executing eecbs script
import pandas as pd  # For smart batch running
import pdb # For debugging
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For utils

def str2bool(v: str) -> bool:
    """Converts a string to a boolean value. Used for argparse."""
    return v.lower() in ("yes", "true", "t", "1")

mapsToMaxNumAgents = {
    "Paris_1_256": 1000, # Verified
    "random_32_32_20": 409, # Verified
    "random-32-32-10": 461, # Verified
    "den520d": 1000, # Verified
    "den312d": 1000, # Verified
    "empty-32-32": 511, # Verified
    "empty-48-48": 1000, # Verified
    "ht_chantry": 1000, # Verified
}

    
    
def runOnSingleInstance(runnerArgs, mapname, numAgents, seed, scenfile):
    """Command for running Python model"""
    # Simulator parameters
    command = "python -m main_pys.simulator"
    for aKey in runnerArgs:
        command += " --{}={}".format(aKey, runnerArgs[aKey])
    
    command += f" --mapNpzFile=data/constant_npzs/all_maps.npz"
    command += f" --mapName={mapname} --scenFile={scenfile} --agentNum={numAgents}"
    command += f" --bdNpzFile=data/constant_npzs/bd_npzs/{mapname}_bds.npz"
    command += f" --seed={seed}"
    print(command)
    subprocess.run(command.split(" "), check=True) # True if want failure error
    
    
def detectExistingStatus(runnerArgs, mapfile, scenfile, aNum, seed, df): # TODO update
    """
    Output:
        If has been run before
        Success if run before
    """
    if isinstance(df, str):
        if not os.path.exists(df):
            return False, 0
        df = pd.read_csv(df, index_col=False)  # index_col=False to avoid adding an extra index column
    # print(df)
    assert(isinstance(df, pd.DataFrame))

    ### Grabs the correct row from the dataframe based on arguments
    for aKey, aValue in runnerArgs.items():
        if aKey in ["outputCSVFile", "mapNpzFile", "timeLimit"]:
            continue
        df = df[df[aKey] == aValue]  # Filter the dataframe to only include the runs with the same parameters
        
    pymodel_map_name = mapfile.split("/")[-1].removesuffix(".map")
    assert(pymodel_map_name in mapsToMaxNumAgents.keys())
    df = df[(df["mapName"] == pymodel_map_name) & (df["scenFile"] == scenfile) & (df["agentNum"] == aNum) & (df["seed"] == seed)]
    
    ### Checks if the corresponding runs in the df have been completed already
    if len(df) > 0:
        if len(df) > 1:
            print("Warning, multiple runs with the same parameters, likely due to a previous crash")
            print("Map: {}, NumAgents: {}, Scen: {}, # Found: {}".format(mapfile, aNum, scenfile, len(df)))
        else:
            success = df["success"].values[0] == 1
        return True, success
    else:
        return False, 0

def runOnSingleMap(runnerArgs, mapName, agentNumbers, seeds, scens):
    outputCSV = runnerArgs["outputCSVFile"]
    for aNum in agentNumbers:
        print("Starting to run {} agents on map {}".format(aNum, mapName))
        numSuccess = 0
        numToRunTotal = len(scens) * len(seeds)
        for scen in scens:
            for seed in seeds:
                runBefore, status = detectExistingStatus(runnerArgs, mapName, scen, aNum, seed, outputCSV)
                if not runBefore:
                    runOnSingleInstance(runnerArgs, mapName, aNum, seed, scen)
                    runBefore, status = detectExistingStatus(runnerArgs, mapName, scen, aNum, seed, outputCSV)
                    assert(runBefore)
                numSuccess += status

        if numSuccess < numToRunTotal/4:
            print("Early terminating as only succeeded {}/{} for {} agents on map {}".format(
                                            numSuccess, numToRunTotal, aNum, mapName))
            break

def helperCreateScens(numScens, mapName, dataPath):
    if mapName == "random_32_32_20":
        mapName = "random-32-32-20"
    scens = []
    for i in range(1, numScens+1):
        scenPath = "{}/mapf-scen-random/{}-random-{}.scen".format(dataPath, mapName, i)
        scens.append(scenPath)
    return scens


"""
python -m main_pys.simple_batch_runner \
      den312d \
      --modelPath=data/model/max_test_acc.pt \
      --maxSteps=100x --seed=0 --useGPU=True \
      --shieldType=Real-Time-LaCAM --lacamLookahead=1
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mapName", help="map name without .map, needs to be in mapsToMaxNumAgents defined in the top", type=str) # Note: Positional is required
    parser.add_argument("--dataPath", help="path to benchmark dataset, should contain mapf-map/ and mapf-scen-random/ folders",
                                      type=str, default="data")
    parser.add_argument("--num_scens", help="number of scenarios to run", type=int, default=25)
    # Simulator parameters
    parser.add_argument('--modelPath', type=str, required=True)
    parser.add_argument('--useGPU', type=lambda x: bool(str2bool(x)), required=True)
    parser.add_argument('--maxSteps', type=str, help="int or [int]x, e.g. 100 or 2x to denote multiplicative factor", required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--shieldType', type=str, default='CS-PIBT', choices=['CS-PIBT', 'CS-Freeze', 'LaCAM', 'Real-Time-LaCAM'])
    parser.add_argument('--lacamLookahead', type=int, help="LaCAM node expansion limit", default=0)
    parser.add_argument('--timeLimit', type=int, help="Time limit (s)", default=60)
    # Output parameters
    parser.add_argument("--logPath", help="path to log folder", type=str, default="logs/") 
    parser.add_argument("--outputCSV", help="outputCSV, ends with .csv", type=str, default="") # Will be saved to logPath+outputCSV
    args = parser.parse_args()

    if args.mapName not in mapsToMaxNumAgents:
        raise KeyError("Map name {} not found in mapsToMaxNumAgents. Please add it to mapsToMaxNumAgents into the top of batch_script.py".format(args.mapName))

    ### Create the folder for the output file if it does not exist
    if args.outputCSV == "":
        args.outputCSV = args.mapName + ".csv"
    totalOutputCSVPath = "{}/{}".format(args.logPath, args.outputCSV)
    if not os.path.exists(os.path.dirname(totalOutputCSVPath)):
        os.makedirs(os.path.dirname(totalOutputCSVPath))

    pymodelArgs = {
        # Map parameters
        "mapNpzFile": "data/constant_npzs/all_maps.npz", 
        "mapName": args.mapName,
        
        # Simulator parameters
        "modelPath": args.modelPath,
        "useGPU": args.useGPU,
        "maxSteps": args.maxSteps,
        "seed": args.seed,
        "shieldType": args.shieldType,
        "lacamLookahead": args.lacamLookahead,
        "timeLimit": args.timeLimit,
        "outputCSVFile": totalOutputCSVPath,
    }
    seeds = [0]
    if args.num_scens > 25:
        raise ValueError("num_scens should be less than or equal to 25")
    scens = helperCreateScens(args.num_scens, args.mapName, args.dataPath)

    increment = mapsToMaxNumAgents[args.mapName] // 10
    agentNumbers = list(range(increment, mapsToMaxNumAgents[args.mapName]+1, increment))

    ### Run model
    runOnSingleMap(pymodelArgs, args.mapName, agentNumbers, seeds, scens)
    
    # Run with CS-PIBT
    pymodelArgs["shieldType"] = "CS-PIBT"
    runOnSingleMap(pymodelArgs, args.mapName, agentNumbers, seeds, scens)