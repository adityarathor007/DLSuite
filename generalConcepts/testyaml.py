import yaml


with open("intro.yaml", "r") as file:
    data = yaml.safe_load(file)  # Safe parsing

# Print the loaded data (Python dictionary)
print(data)

# Accessing values
print("Person Name:", data["person"]["name"])
print("Car Model:", data["car"]["model"])
print("Hobbies:", data["person"]["hobbies"])
