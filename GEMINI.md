# Gemini Guidelines

This document provides guidelines for the Gemini AI assistant to follow when working in this repository.

## Project Overview

This project uses Isaac Sim to simulate a UR10 robot performing a multi-object pick and place task. The robot picks up several cubes from a bin on one table and arranges them in a grid on another table.

## Tech Stack

*   **Primary Language:** Python
*   **Simulation Environment:** Isaac Sim 5.0

## Getting Started

The main entry point for the simulation is `ur10_table_stacking.py`. It can be run with:

```bash
python ur10_table_stacking.py [--task <task_name>]
```

The optional `--task` argument can be either `TableTask2` or `TableTask3`.

## Project Structure and File Purpose

*   `ur10_table_stacking.py`: The main executable file that starts the Isaac Sim application. It dynamically imports and runs a task specified via command-line arguments.

*   `stacking_task.py`: Defines the core `UR10MultiPickPlace` task logic. This class manages the robot, the objects to be picked, the target locations, and the overall state of the stacking task.

*   `table_setup.py`: Contains functions to set up the simulation environment. This includes creating the tables, the robot mounting stand, the picking bin, and placing various objects in the scene.

*   `asset_utils.py`: Provides utility functions for creating and adding primitive shapes (e.g., cubes, cylinders) and USD assets to the simulation scene.

*   `tasks/`: This directory contains individual task definitions that can be run by the main simulator.
    *   `table_task2.py`: Defines `TableTask2`, a scenario where the robot picks up cubes and arranges them in a grid.
    *   `table_task3.py`: Defines `TableTask3`, a scenario where the robot picks up objects and arranges them as discs in a grid.

*   `exts/`: Contains external Isaac Sim extension libraries. **Do not modify files in this directory.**

## AI Assistant Guidelines

*   **Do not modify files in the `exts/` directory.** This directory contains external libraries and should not be altered.
*   **Do not automatically run tests.** The testing suite requires a specific Isaac Sim environment to be running. Please ask before running any tests.
