"""
Entry point for the Medical Diagnosis Bot.

Usage:
    python main.py                          # run with built-in demo patient
    python main.py --patient data/patient.json   # run with a custom JSON file
    python main.py --train                  # retrain the ML model and exit

Environment:
    ANTHROPIC_API_KEY must be set (see .env.example).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

load_dotenv()

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Demo patient cases
# ---------------------------------------------------------------------------

DEMO_PATIENTS = {
    "pneumonia": {
        "age": 65,
        "temperature": 103.2,
        "heart_rate": 108,
        "bp_systolic": 118,
        "bp_diastolic": 76,
        "oxygen_saturation": 91,
        "fever": 1,
        "cough": 1,
        "shortness_of_breath": 1,
        "chest_pain": 1,
        "fatigue": 1,
        "sweating": 1,
    },
    "diabetes": {
        "age": 52,
        "temperature": 98.7,
        "heart_rate": 78,
        "bp_systolic": 138,
        "bp_diastolic": 88,
        "oxygen_saturation": 97,
        "frequent_urination": 1,
        "blurred_vision": 1,
        "fatigue": 1,
        "nausea": 1,
    },
    "covid19": {
        "age": 40,
        "temperature": 100.9,
        "heart_rate": 92,
        "bp_systolic": 122,
        "bp_diastolic": 80,
        "oxygen_saturation": 94,
        "fever": 1,
        "cough": 1,
        "loss_of_taste_smell": 1,
        "fatigue": 1,
        "body_ache": 1,
    },
    "hypertension_crisis": {
        "age": 58,
        "temperature": 98.6,
        "heart_rate": 88,
        "bp_systolic": 195,
        "bp_diastolic": 118,
        "oxygen_saturation": 96,
        "headache": 1,
        "dizziness": 1,
        "chest_pain": 1,
        "blurred_vision": 1,
    },
    "emergency": {
        # This will trigger the EMERGENCY guardrail (SpO2 < 90)
        "age": 72,
        "temperature": 104.5,
        "heart_rate": 130,
        "bp_systolic": 88,
        "bp_diastolic": 55,
        "oxygen_saturation": 82,
        "fever": 1,
        "cough": 1,
        "shortness_of_breath": 1,
        "chest_pain": 1,
        "fatigue": 1,
    },
}


def run_demo(case_name: str = "pneumonia") -> None:
    patient = DEMO_PATIENTS.get(case_name, DEMO_PATIENTS["pneumonia"])
    console.print(Rule(f"[bold cyan]Demo Patient: {case_name}[/bold cyan]"))
    console.print(Panel(json.dumps(patient, indent=2), title="Patient Data"))
    _run_crew(patient)


def run_from_file(filepath: str) -> None:
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]File not found: {filepath}[/red]")
        sys.exit(1)
    with open(path) as f:
        patient = json.load(f)
    console.print(Rule("[bold cyan]Patient from file[/bold cyan]"))
    console.print(Panel(json.dumps(patient, indent=2), title="Patient Data"))
    _run_crew(patient)


def _run_crew(patient: dict) -> None:
    from src.crew import MedicalDiagnosisCrew  # import here to allow --train flag

    console.print("\n[bold yellow]Starting Medical Diagnosis Crew...[/bold yellow]\n")
    try:
        crew = MedicalDiagnosisCrew()
        report = crew.run(patient)
        console.print(Rule("[bold green]Clinical Decision Support Report[/bold green]"))
        console.print(report)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logging.exception("Crew run failed")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medical Diagnosis Bot")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain the disease classifier and exit.",
    )
    parser.add_argument(
        "--patient",
        type=str,
        default=None,
        help="Path to a JSON file containing patient data.",
    )
    parser.add_argument(
        "--demo",
        type=str,
        default="pneumonia",
        choices=list(DEMO_PATIENTS.keys()),
        help="Name of the built-in demo patient case to run.",
    )
    args = parser.parse_args()

    if args.train:
        console.print("[bold]Retraining disease classifier...[/bold]")
        from src.models.disease_classifier import train_and_save
        train_and_save()
        console.print("[green]Training complete.[/green]")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        console.print(
            "[red]ANTHROPIC_API_KEY is not set.[/red] "
            "Copy .env.example to .env and add your key."
        )
        sys.exit(1)

    if args.patient:
        run_from_file(args.patient)
    else:
        run_demo(args.demo)


if __name__ == "__main__":
    main()
