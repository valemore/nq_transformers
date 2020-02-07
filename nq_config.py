# This module stores global variables that are stored using abseil FLAGS in Google's code.

max_context_length = None    # max tokens in context
max_question_length = None   # max tokens in question
max_windows = None           # max windows per question

window_stride = None         # stride for the sliding window approach

predict_batch_size = None    # Batch size to use for evaulation

# Preprocessing flags
skip_nested_contexts = True  # Completely ignore context that are not top level nodes in the page.
max_position = 50            # Maximum context position for which to generate special tokens.
include_unknowns = None      # If positive, probability of including answers of type `UNKNOWN`. -- Inverse of our undersampling factor

# Evaluation flags
n_best_size = None           # The total number of n-best predictions to generate
max_answer_length = None     # The maximum length of an answer that can be generated
