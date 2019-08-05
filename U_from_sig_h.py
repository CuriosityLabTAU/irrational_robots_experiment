import pandas as pd
import general_quantum_operators as gpo

# h_suspects = [-0.9989711556,1,-0.6474543747,0.5060843493,0.5758584677,0.197824478,-0.8921814559,0.2611674383,0.5663847889,0.837203556]


h_suspects = [1,1,1,1,1,1,-1,1,-1,1]
h_art = [0.7740711626,-0.8857690154,-0.9800438384,0.8281932899,-0.7147230029,-0.9896616931,-0.9667894209,-0.6690243139,-0.7320855243,-0.7691979699]

U_suspects = gpo.U_from_H(gpo.grandH_from_x(h_suspects))
U_art = gpo.U_from_H(gpo.grandH_from_x(h_art))

pd.DataFrame(U_suspects).to_csv('U_suspect.csv')
pd.DataFrame(U_art).to_csv('U_art.csv')

# U_suspects = pd.read_csv('U_suspect.csv', index_col=0).applymap(complex)
# U_art = pd.read_csv('U_art.csv', index_col=0).applymap(complex)
#
# print(U.shape)