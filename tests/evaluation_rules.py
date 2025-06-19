def get_test_cases(df):
    return [
        {
            "question": "Which device has the highest average CPU temperature?",
            "expected": lambda df: df.groupby("device_id")["cpu_temp"].mean().idxmax()
        },
        {
            "question": "Which device has the most WARNING statuses?",
            "expected": lambda df: df[df["status"] == "WARNING"]["device_id"].value_counts().idxmax()
        },
        {
            "question": "What is the average bandwidth?",
            "expected": lambda df: round(df["bandwidth"].mean(), 2)
        },
        {
            "question": "What's the average bandwidth of all devices?",
            "expected": lambda df: round(df["bandwidth"].mean(), 2)
        },

    ]
