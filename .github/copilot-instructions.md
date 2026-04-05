# Copilot Instructions for medical-rag-project

## Project Overview
This project aims to develop a trustworthy clinical decision framework with a focus on explainability and safety guardrails. The architecture and implementation details are currently sparse, so contributions should prioritize clarity, modularity, and adherence to best practices for clinical-grade software.

## Key Guidelines

### 1. Code Structure and Organization
- **Modularity**: Ensure that code is organized into small, reusable components. This is critical for maintaining the explainability and safety of the framework.
- **Documentation**: Add clear comments and docstrings to explain the purpose and functionality of each module, especially for algorithms and decision-making logic.

### 2. Testing and Validation
- **Unit Tests**: Write unit tests for all new functionality. Use realistic clinical scenarios to validate decision-making logic.
- **Safety Checks**: Implement and test safety guardrails rigorously. For example, ensure that the system flags ambiguous or unsafe recommendations.

### 3. Explainability
- **Transparent Logic**: Use clear and interpretable algorithms. Avoid "black-box" models unless accompanied by robust explainability techniques.
- **Logging**: Include detailed logs for decision-making processes to aid debugging and auditing.

### 4. External Dependencies
- **Dependency Management**: Use a `requirements.txt` or equivalent to manage dependencies. Ensure that all dependencies are well-documented and vetted for clinical use.

### 5. Collaboration
- **Pull Requests**: Provide detailed descriptions for pull requests, including the problem being solved, the approach taken, and any trade-offs considered.
- **Code Reviews**: Focus on maintainability, clarity, and adherence to safety and explainability principles during code reviews.

## Examples
- **Safety Guardrails**: If implementing a feature that recommends treatments, ensure that the code includes checks for contraindications and flags them appropriately.
- **Explainability**: When adding a new decision-making algorithm, include a module that outputs a human-readable explanation of the decision.

## Key Files and Directories
- `README.md`: High-level project overview. Expand this file as the project evolves.
- Additional directories and files should be documented here as the project structure becomes more defined.

## Future Updates
This document should evolve alongside the project. Contributors are encouraged to update these instructions as new patterns and conventions emerge.