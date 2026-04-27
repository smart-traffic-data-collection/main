# Simple Guide to GitHub Basics

This guide will help you set up an SSH key to connect to GitHub without needing a password every time. It will also show you the basic steps to work with your code.

## 1. Create an SSH Key

An SSH key is like a secret digital handshake between your computer and GitHub.

**Step 1:** Open your computer's terminal.<br/>
**Step 2:** Type this command, replacing the email with your GitHub email, and press Enter:
```
ssh-keygen -t ed25519 -C "your_email@example.com"
```
**Step 3:** When it asks where to save the key, just press **Enter** to accept the default location.<br/>
**Step 4:** When it asks for a passphrase (password), you can just press **Enter** twice to skip it. This makes it easier to use.<br/>
**Step 5:** Now, you need to see your new key so you can copy it. Run this command:
```
cat ~/.ssh/id_ed25519.pub
```

**Step 6:** Copy the text that appears on your screen.<br/>
**Step 7:** Go to your GitHub account on the web:
1. Click your profile picture in the top right and choose **Settings**.
2. Click **SSH and GPG keys** on the left side.
3. Click the green **New SSH key** button.
4. Give it a title (like "My Laptop") and paste your copied text into the "Key" box.
5. Click **Add SSH key**.

---

## 2. Download a Project (Git Clone)
Once your SSH key is set up, you can download a project from GitHub to your computer using your new secure connection. Open your terminal, go to the folder where you want to save the project, and run:
```bash
git clone git@github.com:smart-traffic-data-collection/main.git
```

---

## 3. Daily GitHub Workflow

Here are the basic commands you will use to work on your code. Open your terminal, go into your newly downloaded project folder (e.g., type `cd project-name`), and use these steps:

### Get the newest code
Before you start working, make sure you have the latest updates from your team.
```bash
git fetch
git pull
```

### Make a new branch
A branch is a safe, separate space where you can make changes without breaking the main code.
```bash
git checkout -b your-branch-name
```
*(Replace `your-branch-name` with a short name for what you are working on, like `name/fix-login-button`)*

### Save your files
After you make changes to your files, tell Git to get them ready to be saved. The period (`.`) means "add all the changed files."
```bash
git add .
```

### Write a message for your changes
Now, officially save your changes with a short message explaining what you did.
```bash
git commit -m "added a new commit"
```

### Send your code to GitHub
Finally, upload your new branch and your saved changes to GitHub so others can see them.
```bash
git push