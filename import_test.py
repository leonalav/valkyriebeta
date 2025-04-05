import traceback

try:
    import train_aio
    print("Successfully imported train_aio.py")
except Exception as e:
    print(f"Error importing train_aio.py: {e}")
    traceback.print_exc() 