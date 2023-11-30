import plotly.express as px

def plot_labeled_sequence(
    timeseries,
    true_repetitions,
    other_exercises=[],
    signals=["linAccZ_orig", "linAccZ"],
    time_column_name="time",
    peaks=None,
):
    f = px.line(
        timeseries.reset_index(),
        x=time_column_name,
        y=signals,
        title="Labeled Series",
    )
    if peaks is not None:
        f.add_scatter(x=peaks.ts, y=peaks.value, mode="markers", name="Peak")

    for exercise in true_repetitions:
        f.add_vrect(
            x0=exercise["start"],
            x1=exercise["end"],
            line_width=1,
            line_color="black",
            annotation_text=exercise["event"],
            annotation_position="bottom right",
            annotation_textangle=0,
            opacity=1,
            name="True",
        )

    for exercise in other_exercises:
        f.add_vrect(
            x0=exercise["start"],
            x1=exercise["end"],
            line_width=2,
            line_color="green",
            line_dash="dash",
            annotation_text=exercise["event"],
            annotation_position="top right",
            annotation_textangle=-90,
        )

    f.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return f