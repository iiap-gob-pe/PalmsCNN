import yaml

# Lee el archivo environment.yml
with open('environment.yml', 'r') as file:
    env_data = yaml.safe_load(file)

# Extrae las dependencias
dependencies = env_data.get('dependencies', [])

# Instala las dependencias
for package in dependencies:
    if isinstance(package, str):  # Dependencias de Conda
        if '=' in package:  # Si tiene una versión específica
            !pip install {package}
        else:
            !pip install {package}
    elif isinstance(package, dict) and 'pip' in package:  # Dependencias de Pip
        for pip_package in package['pip']:
            !pip install {pip_package}

