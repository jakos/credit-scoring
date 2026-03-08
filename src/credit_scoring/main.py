from credit_scoring import score_applicant


if __name__ == "__main__":
    example_score = score_applicant(income=5000, debt=1200)
    print(f"Sample score: {example_score}")
