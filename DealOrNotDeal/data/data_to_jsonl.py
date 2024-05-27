import ujson as json
import random


def parse_line(line):
    values = list(map(int, line.split()))
    return {
        "books_value": values[1],
        "hats_value": values[3],
        "balls_value": values[5]
    }


def convert_to_jsonl(file_path, output_file_path, num_pairs, seed=123456):
    # set random seed
    random.seed(seed)

    with open(file_path, 'r') as file:
        lines = file.readlines()

    if len(lines) % 2 != 0:
        lines = lines[:-1]

    # generate random index
    line_indices = list(range(0, len(lines), 2))
    random.shuffle(line_indices)
    selected_indices = line_indices[:num_pairs]

    with open(output_file_path, 'w') as jsonl_file:
        for i in selected_indices:
            alice_line = parse_line(lines[i].strip())
            bob_line = parse_line(lines[i + 1].strip())

            entry = {
                "count": {
                    "books_cnt": int(lines[i].split()[0]),
                    "hats_cnt": int(lines[i].split()[2]),
                    "balls_cnt": int(lines[i].split()[4]),
                },
                "value": {
                    "Alice": alice_line,
                    "Bob": bob_line
                }
            }

            jsonl_file.write(json.dumps(entry) + '\n')


if __name__ == '__main__':
    file_path = 'selfplay.txt'  # Replace with the actual file path
    output_file_path = 'items_demo.jsonl'  # Changed to .jsonl extension
    convert_to_jsonl(file_path, output_file_path, 50)
