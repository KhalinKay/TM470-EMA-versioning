"""15-question test set for the chunk-size comparison (Section 4.3 of the report).

Questions are grounded in the shipped sample documents in data/sample_docs/
(a "History and Philosophy of Religion" corpus: religion_ancient_origins.txt,
abrahamic_traditions.txt, eastern_traditions.txt) spanning factual, definition,
comparison and synthesis question types. Replace these questions if you swap
in your own documents before running evaluation/run_evaluation.py.
"""

TEST_QUESTIONS = [
    {"question": "Who was Akhenaten and what religious reform is he associated with?", "category": "factual"},
    {"question": "What is the significance of the Council of Nicaea in 325 CE?", "category": "factual"},
    {"question": "What are the Four Noble Truths in Buddhism?", "category": "definition"},
    {"question": "How did the Sunni-Shia split within Islam originate?", "category": "factual"},
    {"question": "Compare the concept of the afterlife in ancient Egyptian religion to the Abrahamic traditions.", "category": "comparison"},
    {"question": "What role did the Torah play in early Judaism?", "category": "factual"},
    {"question": "Explain the concept of dharma in Hinduism.", "category": "definition"},
    {"question": "What led to the East-West Schism of 1054?", "category": "factual"},
    {"question": "Summarise the religious practices of prehistoric societies described in the material.", "category": "synthesis"},
    {"question": "How does Confucianism differ from Daoism in its approach to social order?", "category": "comparison"},
    {"question": "What are the Five Pillars of Islam?", "category": "definition"},
    {"question": "What was the Enuma Elish and what does it reveal about Mesopotamian religion?", "category": "factual"},
    {"question": "How did Buddhism spread beyond India, and what role did Ashoka play?", "category": "factual"},
    {"question": "What common themes, if any, connect the three Abrahamic religions according to the material?", "category": "synthesis"},
    {"question": "What is Shinto, and how does it relate to Japanese cultural identity?", "category": "definition"},
]
