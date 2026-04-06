# Automated Warehouse Logistics Exception Handler 

## Introduction
In modern automated fulfillment centers, efficiency isn't just about how fast robots move—it’s about how quickly the system recovers when things go wrong. Most automated systems fail when faced with "exceptions": a robot breaking down in a narrow aisle, a sensor reporting "ghost" inventory, or a sudden shipment delay that cascades into a logistical nightmare.

The Automated Warehouse Logistics Exception Handler is a high-fidelity Mini Reinforcement Learning (RL) environment built on the OpenEnv framework. It simulates a dynamic warehouse control system where an AI agent acts as the "Central Dispatcher.It provides a lightweight yet mathematically rigorous sandbox for agents to resolve cascading logistical failures.

## Project Overview

* 🦾 Focused High-Fidelity Simulation — A lightweight yet detailed environment that simulates specific warehouse exceptions like robot blockages, damaged goods, and shipment delays.

* 📡 Stochasticity & Noise — Features a "Noisy Observation" toggle that obscures robot locations and data, forcing agents to handle partial observability through active polling actions.

* 📊 Multi-Tiered Grading — Moves beyond simple binary success/failure rewards by implementing a continuous progress-based scoring system (0.0 to 1.0).

*  🤖 Agent Agnostic Design — Fully compatible with the OpenEnv specification, making it a plug-and-play sandbox for testing both LLM-based agents and traditional Reinforcement Learning models.

* 🏗️ Modular Architecture — Strictly separates environment "physics" from the "grading logic," ensuring high performance and easy extensibility for new warehouse tasks.
 
## Getting Started

### Local Installation(Setup)
 ####  Prequisites
  Before you begin ensure you have the following installed and verified 

 * Python(3.10+):The core runtime for the environment.
 * Git: To clone the repository.
 * pip:Python's package installer

#### Step 1: Clone the Repository
 Open your terminal or powershell and run:

    git remote add origin "https://github.com/samarth-2006SJW/MINI_RL_ENVIRONMENT.git"

 Navigate to MINI_RL_ENVIRONMENT folder by running:

    cd MINI_RL_ENVIRONMENT

#### Step 2: Create a virtual environment and activation
In powershell inside MINI_RL_ENVIRONMENT run:

    python -m venv .venv
    .venv/Scripts/activate

For macOS/Linux users:

    python3 -m venv venv
    source venv/bin/activate

#### Step 3: Installing Dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

* This will install essential libraries including fastapi, uvicorn, openenv-core, and pydantic.

#### Step 4:Running the application
Firstly you'll need your own API keys,LLM models and their Base URL's.

* Inside powershell run:

 `#Set your OpenAI API Key (or compatible provider like NVIDIA NIM)`

    $env:OPENAI_API_KEY="paste-your-own-key"
` #Set the base URL for the LLM provider` 

    $env:LLM_BASE_URL="paste-your-own-URL"

` #Specify the model you want the agent to use`

    $env:LLM_MODEL="paste-model-you-use"

* For macOS/Linux users run:

`# Set your OpenAI API Key`

    export OPENAI_API_KEY="your-api-key-here"

`# Set the base URL for the LLM provider`

    export LLM_BASE_URL="https://api.openai.com/v1"

`# Specify the model you want the agent to use`

    export LLM_MODEL="gpt-4o"
* Now run:

      python app.py

## System Architecture & Logical Flow 
```mermaid
 flowchart LR
    %% Custom UI Colors
    classDef blockNode fill:#2d3436,stroke:#74b9ff,stroke-width:2px,color:#dfe6e9,rx:8px,ry:8px;
    classDef dataNode fill:#2d3436,stroke:#55efc4,stroke-width:2px,color:#dfe6e9;
    classDef actorNode fill:#d63031,stroke:#ff7675,stroke-width:2px,color:#ffffff,shape:circle;

    %% 1. Force Agent as the absolute FIRST component on the Left
    subgraph Trigger [User Space]
        Agent((User / Agent)):::actorNode
    end

    %% 2. Presentation Layer cleanly side-by-side
    subgraph Presentation [API & UI Layer]
        App["app.py"]:::blockNode
    end

    %% 3. Core system side-by-side
    subgraph Core [Core Physics & Logic]
        Env["environment.py"]:::blockNode
        Models["models.py"]:::blockNode
        Utils["utils.py"]:::blockNode
        Config[("openenv.yaml")]:::dataNode
    end

    %% 4. Task Check side-by-side
    subgraph Eval [Task Validation]
        Tasks["tasks.py"]:::blockNode
    end

    %% Forward Flow (Engine routes automatically left to right)
    Agent -->|1. Takes Action| App
    App -->|2. Step / Reset| Env
    
    Env -.->|3. Data Schemas| Models
    Env -.->|4. Helpers Script| Utils
    Env -->|5. Load Params| Config
    
    Env -->|6. Check Status| Tasks
    
    %% Return Feedback Flow (Engine wraps arrows around boxes neatly)
    Tasks -->|7. Evaluated Reward| Env
    Env -->|8. Observation Array| App
    App -->|9. Update UI & JSON| Agent


```    
## Core Mission

 Unlike standard pathfinding simulations, this project focuses on Decision Intelligence under Uncertainty. The agent is tasked with:

* Triage & Resolution: Identifying and fixing critical failures (Robot blockages, inventory shortages, shipment delays).

* Handling Partial Observability: Managing "Noisy Observations" where sensors may fail or provide incomplete data, requiring the agent to proactively "poll" for better information.

* Multi-Objective Optimization: Balancing speed of resolution against resource costs and operational penalties.

By utilizing a modular architecture—separating "The Physics" (environment logic) from "The Judge" (grading logic)—this project provides a rigorous testing ground for LLM-based agents and traditional RL models to prove they can handle the chaotic edge cases of real-world logistics.

