import pandas as pd
import general_quantum_operators as gpo

h = [-0.9989711556,1,-0.6474543747,0.5060843493,0.5758584677,0.197824478,-0.8921814559,0.2611674383,0.5663847889,0.837203556]

U_suspects = gpo.U_from_H(gpo.grandH_from_x(h))

pd.DataFrame(U_suspects).to_csv('U_suspect.csv')