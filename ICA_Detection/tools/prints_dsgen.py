import platform
import os

def get_system_details():
    """Gathers basic system details."""
    details = {
        "System": platform.system(),
        "Node Name": platform.node(),
        "Release": platform.release(),
        "Version": platform.version(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "Python Version": platform.python_version(),
        "User": os.getlogin() if hasattr(os, 'getlogin') else 'N/A'
    }
    return details

def print_welcome_and_structure():
    """
    Prints a welcome message, system details, and describes the expected folder structure
    for the Coronary Angiography Detection project.
    """

    project_name = "Coronary Angiography Detection"
    # Define authors as a list of strings
    authors_list = [
        "Mario Pascual-González", # Placeholder
        "Ariadna Jiménez-Partinen", # Placeholder
        "Esteban J. Palomo", 
        "Ezequiel López-Rubio",
        "Almudena Ortega-Gómez",
    ]
    authors_display = ", ".join(authors_list) if authors_list else "[Authors Not Specified]"

    ascii_art_welcome = r"""
    

                  .=******=    -****-                      
            :***:   =*   =    :       ****                 
         =*:  ***     :   :**      +*  =  *:*              
       *=   :    :    *        *.          +  *-           
     * *   ::    *    +        *    :=*=:       *=         
   =*:  :=*        *  *=    -**=    *       -   =**.       
  *-   :  +***=     +     **   =**+.         *     =*      
 *: *    =*          -   .*       *          *       *     
.=+.  +           ***    *             *     =  ==  ***    
* :  =      =+*=:    .   *      *     +     .   *  *   =   
*=     =  .=       =      *   :*=.:-=*      *          .*  
*+  *   *+        +        :*         .    * +      - +  * 
:*      *      *=**=**++***               *    .++- *  * *.
 *   :    =+.   =*:    .= .-           ==*       .  : .  =+
  *...*.      **     . *     .*          *         +   - **
   =***  =****     *   +      .**    -:        *          *
       -*****:    =     *    .:           =:             *:
             *    *  .   :  .   .              .        :* 
             **   =       .*: *   =   .*      *:    *   *  
             ** :  =.     *    :=-==+*****:  :+**=  .+*:   
              :**  .     .  :**** ..- -*****:  -*.*        
                   *******- = * +.*-**=*  *  + :**         
                             .***** :::*  =*+***.          
                                =*  ==*                    
                                  *-  .+                   
                                    *   *                  
                                     ***:                  

__/\\\\\\\\\\\________/\\\\\\\\\_____/\\\\\\\\\_____/\\\\\\\\\\\_        
 _\/////\\\///______/\\\////////____/\\\\\\\\\\\\\__\/////\\\///__       
  _____\/\\\_______/\\\/____________/\\\/////////\\\_____\/\\\_____      
   _____\/\\\______/\\\_____________\/\\\_______\/\\\_____\/\\\_____     
    _____\/\\\_____\/\\\_____________\/\\\\\\\\\\\\\\\_____\/\\\_____    
     _____\/\\\_____\//\\\____________\/\\\/////////\\\_____\/\\\_____   
      _____\/\\\______\///\\\__________\/\\\_______\/\\\_____\/\\\_____  
       __/\\\\\\\\\\\____\////\\\\\\\\\_\/\\\_______\/\\\__/\\\\\\\\\\\_ 
        _\///////////________\/////////__\///________\///__\///////////__
    """
    paper_title = "Bayesian Hyperparameter Optimization of YOLO Models for Invasive Coronary Angiography Lesion Detection and Assessment"  # Placeholder
    journal = "Computers In Biology and Medicine"  # Placeholder

    system_info = get_system_details()

    welcome_message = f"""
    ===================================================================================
    {ascii_art_welcome}

    Welcome to the {project_name} project!

    Authors: {authors_display}
    Paper: {paper_title}
    Journal: {journal}

    This script will help you generate and preprocess datasets for your project.
    Please ensure your data is organized according to the expected structure.

    System Information:
    -------------------
    User: {system_info['User']}
    System: {system_info['System']} {system_info['Release']} ({system_info['Version']})
    Machine: {system_info['Machine']}
    Processor: {system_info['Processor']}
    Python Version: {system_info['Python Version']}
    Node: {system_info['Node Name']}

    ===================================================================================
    """
    print(welcome_message)

    folder_structure_description = """
    Expected Folder Structure:
    --------------------------

    Your project should be organized as follows:

    / (Your Project Root)
    |-- /path/to/source_datasets/ (Defined as 'root_dir_source_datasets' in config.yaml)
    |   |-- CADICA/
    |   |-- ARCADE/
    |   |-- KEMEROVO/
    |
    |-- /path/to/output_folder/ (Defined as 'output_folder' in config.yaml)
    |   |-- stenosis_detection/
    |   |   |-- images/            (Processed images for detection)
    |   |   |-- labels/            (Processed labels for detection)
    |   |   |-- json/
    |   |   |   |-- combined_standardized.json
    |   |   |   |-- planned_standardized.json
    |   |   |   |-- splits.json
    |   |
    |   |-- arteries_segmentation/
    |   |   |-- images/            (Processed images for segmentation)
    |   |   |-- labels/            (Processed labels for segmentation)
    |   |   |-- json/
    |   |   |   |-- combined_standardized.json
    |   |   |   |-- planned_standardized.json
    |   |   |   |-- splits.json

    ===================================================================================
    """
    print(folder_structure_description)

if __name__ == "__main__":
    # This allows you to run prints.py directly to see the output
    print_welcome_and_structure()