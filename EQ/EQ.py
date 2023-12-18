def calculate_emotional_quotient():
    total_questions = 0
    correct_answers = 0

    # Emotional awareness
    questions = [
        ("Which of the following emotions do I have the least experience with?", "Happiness"),
        ("When someone I know is going through a hard time, which emotion should I focus on trying to help them feel better?", "Compassion"),
        ("What emotion would someone experience if they just found out that their house burned down?", "Sadness"),
    ]

    for question, correct_answer in questions:
        total_questions += 1
        answer = input(f"Question {total_questions}: {question} ")
        if answer.lower() == correct_answer.lower():
            correct_answers += 1

    # Calculate EQ score
    score = (correct_answers / total_questions) * 100
    print(f"Your Emotional Quotient score is {score}%")

calculate_emotional_quotient()