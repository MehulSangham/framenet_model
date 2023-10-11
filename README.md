# Frame Detection using FrameNet                                                                                                                                             
                                                                                                                                                                                
   This project is about identifying frames within a corpus using FrameNet data. The functionality is encapsulated in three Python scripts that parse the data, train the       
   model, and identify frames.                                                                                                                                                  
                                                                                                                                                                                
   ## Scripts                                                                                                                                                                   
   - **ParseData.py**: This script contains the `parse_data` function which handles parsing the corpus and FrameNet data.                                                       
                                                                                                                                                                                
   - **TrainModel.py**: This script includes the `train_model` function which handles training a model to detect frames in the corpus.                                          
                                                                                                                                                                                
   - **IdentifyFrames.py**: This script contains the `identify_frames` function which utilizes the trained model to identify frames within a new piece of text.                 
                                                                                                                                                                                
   ## Work Flow                                                                                                                                                                 
   1. Run ParseData.py to parse the corpus and FrameNet data.                                                                                                                   
   2. Use the parsed data as input to the `train_model` function in TrainModel.py which returns a trained model.                                                                
   3. Apply the trained model to a new piece of text using the `identify_frames` function in IdentifyFrames.py to detect frames.                                                
                                                                                                                                                                                
   The actual implementation of these functions are still placeholders and need to be updated based on the project's requirement.