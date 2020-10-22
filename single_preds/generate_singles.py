base_file = 'common.yml'

exp_names = {
    'becertain': 'BeCertain',
    'belpart': 'BelPart',
    'allopen': 'AllOpen',
    'wondowless': 'WondowLess',
}

with open(base_file, 'r') as f:
    the_file = f.read()

for name in exp_names:
    new_file = f'{name}.yml'
    updated_file = the_file.replace('NAME', f'single/{name}')
    updated_file = updated_file.replace('VERB', exp_names[name])
    with open(new_file, 'w') as f:
        f.write(updated_file)