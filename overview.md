# Overview
## Explenation
This file gives a short overview of what needs to be done in the project and our questions


## Done till now
-  Imported SIMPL model into project
- Added specific config file
    - Changed from python class to YAML
- Started overwriting forward function
    - Needs to be written to fit basemodel
    - Should fit what basemodel expects
- Copied in loss function from SIMPL
- Overwritten optimizer s.t matches with SIMPL



## TODO

- Understand both dataset structures
    - S.t can transfer from one to another
    - Want model to handle data conversion if possible
    - Worst case overwrite SIMPLDataset based on Argoverse
- Generate RPE in forward/training_step function
- Check GPU correct
- Check config correct
    - Could be overlap between SIMPL config and unitraj config



## Questions
1. Can we get all the data we need from the batch dictionary? 
2. If not how much do we need to change
3. What are the features in the SIMPL dictionary


## Found out so far
- 