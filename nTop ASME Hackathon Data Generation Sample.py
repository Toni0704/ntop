#Imports
import os
import subprocess 
import json 
import csv
import numpy as np

#Assuming this script, ntop file, and json files will be in the same folder
Current_Directory = os.path.dirname(os.path.realpath('__file__')) 
exePath = r"C:/Program Files/nTopology/nTopology/nTopCL.exe"  #nTopCL path
nTopFilePath = r"nTop_ASME_Hackathon_HEX.ntop"   #nTop notebook file name (should be in the same directory as this file)
Input_File_Name = "input.json"      #JSON input file name to be saved as
Output_File_Name = "output.json"       #JSON output file name to be saved as

#Input variables in JSON structure
Inputs_JSON = {
    "description": "",
    "inputs": [
        {
            "description": "",
            "name": "Cell Size X",
            "type": "real",
            "units": "mm",
            "value": 25.0
        },
        {
            "description": "",
            "name": "Cell Size Y/Z",
            "type": "real",
            "units": "mm",
            "value": 25.0
        },
        {
            "description": "",
            "name": "Inlet Velocity",
            "type": "real",
            "units": "mm*s^-1",
            "value": 3000.0
        }
    ],
    "title": "Simple Heat Exchanger"
}

#nTopCL arguments in a list
Arguments = [exePath]               #nTopCL path
Arguments.append("-j")              #json input argument
Arguments.append(Input_File_Name)   #json path
Arguments.append("-o")              #output argument
Arguments.append(Output_File_Name)  #output json path
Arguments.append(nTopFilePath)      #.ntop notebook file path

#initialize iteration counter for data generation loops
iteration = 0
#define file path name for CSV to store data
csv_file_path = "nTop ASME Hackathon Data.csv"

#use nested for loops to iterate through parameters
for xSize in np.linspace(10,25,5):
    for yzSize in np.linspace(10,25,5):
        for inletVelocity in np.linspace(2500,3500,5):
            #set the input parameters in the JSON
            Inputs_JSON['inputs'][0]['value'] = float(xSize)
            Inputs_JSON['inputs'][1]['value'] = float(yzSize)
            Inputs_JSON['inputs'][2]['value'] = float(inletVelocity)
            print(Inputs_JSON)
            #Creating in.json file
            with open(Input_File_Name, 'w') as outfile:
                json.dump(Inputs_JSON, outfile, indent=4)

            #nTopCL call with arguments
            print(" ".join(Arguments))
            output,error = subprocess.Popen(Arguments,stdout = subprocess.PIPE, 
                        stderr= subprocess.PIPE).communicate()

            #Print the return messages (optional)
            print(output.decode("utf-8"))

            #Read the JSON from the output file and parse just the data
            with open("output.json", 'r') as output:
                data = json.load(output)
                data = data[0]['value']['jsonObject']
                print(data)

            #Define the headers as the data keys for use in the first iteration
            headers = data.keys()
            
            if iteration == 0:
                #Write the file on the first iteration and add a header row first
                with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                    writer.writerow(data.values())
            else:
                #For subsequent iterations, append the values to the CSV
                with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(data.values())
            
            #Increment the iteration counter
            iteration += 1
