import pickle
from pathlib import path 

import streamlit_authenticator as stauth

names = ["Sourav Verdodia" , "Sahil" , "Saurabh Sinha"]
usernames = ["souraverdodia" , "sahiloraon" , "saurabhsinha"]
passwords = ["souraverdodia" , "sahiloraon" , "saurabhsinha"]

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file: 
  pickle.dump(hashed_passwords , file)
  
