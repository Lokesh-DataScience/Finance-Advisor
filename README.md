### 🔧 Installing and Using qpdf to Unlock a Password-Protected PDF

Follow the steps below to install qpdf and decrypt a protected PDF file.

- **1. Install qpdf using Winget**

Run the following command in PowerShell or Command Prompt:

`winget install qpdf`


This installs the qpdf utility on your system.

- **2. Decrypt a Password-Protected PDF**

Once installed, use the command below to remove the password from a PDF:

`qpdf --password=YOUR_PASSWORD --decrypt input.pdf unlocked.pdf`


YOUR_PASSWORD → the PDF’s password

input.pdf → the original locked PDF

unlocked.pdf → the output PDF without password protection