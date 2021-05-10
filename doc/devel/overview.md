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
