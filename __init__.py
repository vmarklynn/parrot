import os

print ("Initializing Speech To Text folder: " + os.getcwd())
if (os.path.exists("parrot/services.py")):
    from . import services
else:
    print("Services file does not exist")
    
