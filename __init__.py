import os

print ("Initializing Speech To Text folder: " + os.getcwd())
if (os.path.exists("parrot/services.py") and os.path.exists("parrot/transcribe_service.py")):
    # from . import services
    from . import transcribe_service
else:
    print("Services file does not exist")
    
