import os

def turn_off_wifi():
    try:
        os.system("netsh wlan disconnect")
        return "Wi-Fi turned off—disconnected from the network!"
    except Exception as e:
        return f"Oops, couldn’t turn off Wi-Fi. Error: {e}"

def open_file_manager():
    try:
        os.system("explorer")
        return "File Manager opened!"
    except Exception as e:
        return f"Oops, couldn’t open File Manager. Error: {e}"

def open_linkedin():
    try:
        os.system("start https://linkedin.com")  # Opens LinkedIn in default browser
        return "LinkedIn opened in your browser!"
    except Exception as e:
        return f"Oops, couldn’t open LinkedIn. Error: {e}"