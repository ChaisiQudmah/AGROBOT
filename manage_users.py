import json
import hashlib
import os

USERS_FILE = "users.json"

def hash_password(password: str) -> str:
    """Hash password with SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users_db):
    with open(USERS_FILE, "w") as f:
        json.dump(users_db, f, indent=4)

def add_user(username: str, password: str, role: str):
    users_db = load_users()
    if username in users_db:
        print(f"⚠️ User '{username}' already exists!")
        return
    users_db[username] = {
        "password": hash_password(password),
        "role": role
    }
    save_users(users_db)
    print(f"✅ User '{username}' added successfully as {role}")

def list_users():
    users_db = load_users()
    if not users_db:
        print("⚠️ No users found.")
    else:
        print("👤 Current Users:")
        for user, details in users_db.items():
            print(f"  - {user} (role: {details['role']})")

def delete_user(username: str):
    users_db = load_users()
    if username not in users_db:
        print(f"⚠️ User '{username}' not found.")
        return
    del users_db[username]
    save_users(users_db)
    print(f"🗑️ User '{username}' deleted successfully.")

if __name__ == "__main__":
    print("\n🔐 AgroBot User Manager")
    print("1. Add user")
    print("2. List users")
    print("3. Delete user")
    choice = input("Choose option: ")

    if choice == "1":
        uname = input("Enter username: ")
        pwd = input("Enter password: ")
        role = input("Enter role (farmer/admin): ")
        add_user(uname, pwd, role)
    elif choice == "2":
        list_users()
    elif choice == "3":
        uname = input("Enter username to delete: ")
        delete_user(uname)
    else:
        print("❌ Invalid choice")
