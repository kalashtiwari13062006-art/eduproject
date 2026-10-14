import joblib
import pandas as pd

from pathlib import Path
from django.conf import settings
from django.shortcuts import render

MODEL_PATH = Path(settings.BASE_DIR) / "saved_model.pkl"
model = joblib.load(MODEL_PATH)


def home(request):
    result = None
    student_name = ""
    roll_number = ""
    error = None

    if request.method == "POST":
        student_name = request.POST.get(
            "student_name", ""
        ).strip()

        roll_number = request.POST.get(
            "roll_number", ""
        ).strip()

        try:
            studytime = int(request.POST.get("studytime", ""))
            failures = int(request.POST.get("failures", ""))
            absences = int(request.POST.get("absences", ""))
            g1 = int(request.POST.get("g1", ""))
            g2 = int(request.POST.get("g2", ""))

            if not student_name or not roll_number:
                raise ValueError(
                    "Please enter student name and roll number."
                )

            if not 1 <= studytime <= 4:
                raise ValueError("Select a valid study time.")

            if not 0 <= failures <= 4:
                raise ValueError("Failures must be between 0 and 4.")

            if not 0 <= absences <= 93:
                raise ValueError("Absences must be between 0 and 93.")

            if not 0 <= g1 <= 20 or not 0 <= g2 <= 20:
                raise ValueError("G1 and G2 must be between 0 and 20.")

            student_data = pd.DataFrame(
                [[studytime, failures, absences, g1, g2]],
                columns=[
                    "studytime",
                    "failures",
                    "absences",
                    "G1",
                    "G2",
                ],
            )

            prediction = model.predict(student_data)[0]
            result = round(max(0, min(20, float(prediction))), 2)

        except (ValueError, TypeError):
            error = (
                "Please enter valid details in all fields "
                "and check the allowed ranges."
            )

    return render(
        request,
        "home.html",
        {
            "result": result,
            "student_name": student_name,
            "roll_number": roll_number,
            "error": error,
        },
    )