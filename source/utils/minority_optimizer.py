# def count_samples_per_class(data):
#     class_counts = [0] * 9  # Assuming 9 classes
#     for _, samples in data.items():
#         for sample in samples:
#             class_id = int(sample[-2])  # class label is the second-to-last element
#             class_counts[class_id] += 1
#     return class_counts


def my_count_samples_per_class(data):
    class_counts = {}
    for _, samples in data.items():
        for sample in samples:
            parts = sample.strip().split(",")
            class_id = int(float(parts[6]))

            if class_id not in class_counts:
                class_counts[class_id] = 0

            class_counts[class_id] += 1
    return class_counts


def find_max(classes):
    classes_count = my_count_samples_per_class(classes)
    max_class = max(classes_count)
    return max_class, classes_count


def minority(p, classes, n):
    n_maxclass, classes_count = find_max(classes)
    mean_samples = float(len(classes) / n)
    alpha = mean_samples / n_maxclass
    rare_classes = []

    for index, each_class in classes_count.items():
        if each_class < (n_maxclass * alpha):
            rare_classes.append(index)
            print(f"Rare class : {index}, samples of class : {each_class}")

    min_thresh = 1
    for each_class_index in rare_classes:
        for _, samples in classes.items():
            for sample in samples:
                parts = sample.strip().split(",")
                class_id = int(float(parts[6]))
                score = float(parts[7])
                if class_id != each_class_index:
                    continue
                if score < min_thresh:
                    min_thresh = score

    return max(min_thresh, p)
