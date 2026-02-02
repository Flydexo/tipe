CONSTRAINTS_LENGTH=29
N_QUERY=12
T=8
state_map = {"q0":1,"q1":2}
caracter_map = {'a': 3, 'b': 5}
transition_map = {("q0", 'a'): 7, ("q0", 'b'):9, ("q1", 'a'):11,("q1", 'b'):13 }
states = {'q0', 'q1'}
alphabet = {'a', 'b'}
transition = {
    'q0': {'a': 'q1', 'b': 'q0'},
    'q1': {'a': 'q0', 'b': 'q1'}
}
start_state = 'q0'
accept_states = {'q0'}
r_f = 8
r_p = 57
blowup_factor = 8