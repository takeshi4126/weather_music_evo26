# Source folder

## Structure

The weather sonification system was developed in 3 phases, and each phase has its own folder as below.

- phase1 folder contains the reproduction of the baseline work, stellar sonification ML model.
- phase2 folder contains the conversion of the baseline work to the weather sonification model.
- phase3 folder contains the incorporation of weather-music association in the latent emotion space, to better communicate the weather pattern.  
- magenta-docker folder contains the docker configuration to run the MelodyRNN used by the phase3.

See the README files inside the individual folders for more details.

## Execution

See README.md files under phase1, phase2 and phase3. Phase3 scripts cannot run before running phase2 scripts. Phase2 scripts should be runnable without running phase1 scripts. 
