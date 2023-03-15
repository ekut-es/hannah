<!--
Copyright (c) 2023 Hannah contributors.

This file is part of hannah.
See https://github.com/ekut-es/hannah for further info.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Development Guidelines

## Code Style

This project uses pre commit hooks for auto formatting and static code analysis.
To enable pre commit hooks run the following command in a `poetry shell`.

     pre-commit install



## Naming Conventions

Try to follow (pep8)[https://pep8.org/#naming-conventions] and the rest of pep8 to the
best of your abilities.

The short summary of pep8 is:

- Use all lowercase names with underscores for methods/functions/modules/packages.
- Use CamelCase for Classes and Type Variables
- Use all upperscore with underscores for constants

## Logging / Console Messages

Try to avoid `print` for doing outputs whenever possible. Use python module (logging)[https://docs.python.org/3.8/howto/logging.html] instead.

Default logglevel for informational messages should be `logging.info()`. If you want to be able to have more finegrained control, use a
dedicated logger for your module e.g.:

    logger = logging.getLogger('datasets.speech')
    ...
    logger.info()
    logger.debug()


## Testing / CI

Try to provide unit tests for new features, unit tests should be written using [pytest](https://docs.pytest.org). Unit test should be designed to run pretty fast and are run for each push.

Tests marked with ``@pytest.mark.integration` are only triggered when `--integration` is given on the pytest commandline. These tests are only run on pushs to the main branch.



## Resolving merge conflicts in `poetry.lock`

If you have changed `poetry.lock` this can result in merge conflicts.

The easiest way to resolve them is:

```
git checkout --theirs poetry.lock
poetry lock --no-update
```

Try to avoid running `poetry update` on feature branches.