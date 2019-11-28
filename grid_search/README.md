# Grid-Search Explorer
 
Grid-Search Explorer is a tool to automatically perform grid-searches based on user defined variables and constants. 
 
## Features 
 
- Dialog based parameter selection
- Dependency based elimination of redundant variations
- Monitoring of jobs
- Visualization of results

## Performance Optimizations
Beside the implemented features, the following optimizations come with Grid-Search Explorer:
- Central key-value cache with Redis backend
- Migration from cache-probability to number of cached variants per sample

## Prerequisites

- ### Common Setup
    Follow the instructions in root README.md. Be sure you are in a proper Python environment.

- ### Python modules
    You have to install via pip
    - GPUtil 1.4.0
    - psutil 5.5.0
    - redis  latest
    - plotly 4.1.1 (Only for visualization)



- ### Redis
    - Determine the lastest Redis NoSQL Server version from https://redis.io/download
    - Compile Redis and run redis-server
      ```console
      $ wget http://download.redis.io/releases/redis-x.x.x.tar.gz
      $ tar xzf redis-x.x.x.tar.gz
      $ cd redis-x.x.x
      $ make
      $ src/redis-server
      ```
      By default, Redis stores the database in certain intervals on disk. If this is not wanted, please pass a fitting configuration file to the server. 

    - To monitor Redis performance run
      ```console
      $ src/redis-cli --stat
      ```

## Operation
- ### Running Grid-Search Explorer
    - **Please start the Redis-Server first or errors will occur**
    - To run Grid-Search Explorer execute the following command in speech recognition's root path:
      ```console
      $ python -m grid_search.explorer --model <model> --experiment_id <experiment_id>
      ```
- ### Configuration
    Grid-Search Explorer will guide you through the configuration steps. Nevertheless, it is possible to manually edit configuration files. They are located under `<speech_recognition_root>/grid_search/config`
    - #### Configuration Principles
        - ##### Model-independent configuration files
            - `general_exclude.lst`
                Contains a list of settings, which are irrelevant with regard to grid-search variations for all types of models

            -  `general_settings.conf`
                Contains general setting like python version, etc.

        - ##### Model-depended configuration files
            -  `<model_name>_exclude.lst`
                Contains a list of settings, which shall be fixed with regard to variation in the grid-seach

            -  `<model_name>.val` 
                Contains the default values for settings and the wanted variation parameters

        - ##### Folder `classes`
            Contains the structure and hierarchy of dependency based settings. In general, there is no need to touch this when setting up a grid-search experiment.

        - ##### About fixed and variable hyperparameters
            There are two ways of making variables fixed: Adding them to the list of excluded hyperparameters or assigning them a constant value. For clarity, it is recommended to explicitly adding fixed hyperparameters to the model-depended exclude list. If a value is excluded from variation, the default / first value defined in `<model_name>.val` is used.

    - #### Syntax of `<model_name>.val`
        - ##### Start-Stop-Step Configuration for numerical values
            The idea behind start-stop-step is to variate a numerical value from a start value to a stop value over n steps. An entry is built up the following way:

            `<hyperparameter name>;<int|float>;<start value>;<stop value>;<number of steps>`

        - ##### Start-Stop-Step Configuration for numerical lists
            Same as Start-Stop-Step Configuration, but with a fixed count of multiple (count=n) variables, aggregated in a list. An entry is built up the following way:
            
            `<hyperparameter name>;list;<start value #1>;<stop value #1>;<number of steps #1>;...;<start value #n>;<stop value #n>;<number of steps #n>`

        - ##### Predefined values for numerical lists
            This entry type represents simply a static list with n values. Syntax:
            `<hyperparameter name>;<predefined_int|predefined_float>;<value #1>;...;<value #n>`

        - ##### Default value for string-based settings
            Syntax:
            `<hyperparameter name>;str;<default_value>`

        - ##### Default value for boolean-based settings
            Syntax:
            `<hyperparameter name>;bool;<0|1>`

- ### Showing graphical results
    - To show visual results as a multi-axis plot, execute the following command in speech recognition's root path:
      ```console
      $ python -m visualize.visualize --model <model> --experiment_id <experiment_id> (--top_n_accuracy <top_n_accuracy>)
      ```
      Please note, that an axis, that has equal values for all variations, is dropped from the graph for the sake of clarity.

    - You have to have a browser installed on your system to see the results. If you have a non-graphical system, please copy the experiment folder from `<speech_recognition_root>/trained_models/<experiment_id>` to the `trained_models` folder of another machine with graphical support.






