from huggingface_hub import login

def authenticate():
    token = input("Enter your Hugging Face token: ")
    login(token)

if __name__ == "__main__":
    authenticate()
    print('Login successfully completed.')