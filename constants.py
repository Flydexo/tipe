CONSTRAINTS_LENGTH=29
N_QUERY=3
STATE_EVAL_LENGTH=2048
T=8
PTRACE_DOMAIN_SIZE=256
state_map = {"q0":1,"q1":2}
caracter_map = {'a': 3, 'b': 5}
transition_map = {("q0", 'a'): 7, ("q0", 'b'):9, ("q1", 'a'):11,("q1", 'b'):13 }
POSEIDON_TRACE_EVAL_LENGTH=131
EVAL_DOMAIN_LENGTH=2048
AUTOMATA_TRACE_LENGTH=14
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