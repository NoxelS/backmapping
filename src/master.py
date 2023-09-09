from email.message import EmailMessage
from plot_cluster_hist import plot_cluster_hist
import datetime
import smtplib
import socket
import sys
import os

##### CONFIGURATION #####

# data_prefix = "/data/users/noel/data/"        # For smaug
# data_prefix = "/localdisk/noel/"              # For fluffy
data_prefix = "data/"                           # For local

PORT = 10001
ENCODING_STOP_SYMBOL = ":"
ENCODING_FINISHED_SYMBOL = "q"
ENCODING_START_SYMBOL = "s"

# Email Settings
EMAIL_SERVER = "alef"
EMAIL_TARGET = "noel@schwabenland.ch" # This is the email address that will receive the training history plots
EMAIL_USER = 'noel' # The user that is sending the email

##### MASTER SOCKER #####

# The master socket is used to communicate with the child processes that are training the indovidual models.
# After every epoch the child process will send the epoch number and the validation loss to the master process. 
# The master process will then plot the training history of all models and send updated plots via email.
# The arguments are <number of models>

def sendUpdateMail(finished_models: list, number_of_models: int):
    """
    This function will send an email to the user with the current training history plot.
    
    Args:
        finished_models (list): A list of all model names that finished training.
        number_of_models (int): The total number of models that were trained.
    """

    msg = EmailMessage()
    msg['Subject'] = f"Done: {(len(finished_models)/number_of_models)*100:.2f}%"
    msg['From'] = f"Slurm Master 🤖 <{EMAIL_USER}@lusi.uni-sb.de>"
    msg['To'] = EMAIL_TARGET
    msg.set_content(f"{len(finished_models)}/{number_of_models} models finished training")

    # Check if the history image exists
    if os.path.exists(os.path.join("training_history.png")):
        with open("training_history.png", 'rb') as fp:
            img_data = fp.read()
        msg.add_attachment(img_data, maintype='image', subtype='png', filename="training_history.png")
        
    text_part, attachment_part = msg.iter_parts()
    text_part.add_alternative(f"<h1>Training History Update ({len(finished_models)}/{number_of_models} models finished training)</h1><img src='cid:training_history.png'>", subtype='html')

    # Send the message via our own SMTP server.
    s = smtplib.SMTP(EMAIL_SERVER)
    s.send_message(msg)
    s.quit()
    
    print(f"{ts()}: Update email sent")
    
def sendStartMail():
    """
    This function will send an email to the user and notify him that the training has started.
    """
    msg = EmailMessage()
    msg['Subject'] = "Training has started!"
    msg['From'] = f"Slurm Master 🤖 <{EMAIL_USER}@lusi.uni-sb.de>"
    msg['To'] = EMAIL_TARGET
    msg.set_content(f"The training has started!")

    # Send the message via our own SMTP server.
    s = smtplib.SMTP(EMAIL_SERVER)
    s.send_message(msg)
    s.quit()
    
    print(f"{ts()}: Starting email sent")
    
def decode_data(datastring: str) -> tuple:
    """
    Decode the datastring that is sent from the child process to the master process.

    Args:
        datastring (str): The datastring that is sent from the child process to the master process.
    
    Returns:
        tuple: The atom name, epoch and loss of the model.
    """
    if datastring.split(ENCODING_STOP_SYMBOL).__len__() != 3:
        raise Exception(f"Invalid datastring: '{datastring}'")
    
    atom_name, epoch, loss = datastring.split(ENCODING_STOP_SYMBOL)
    
    return atom_name.strip(), int(epoch), float(loss)

def decode_finished(datastring: str) -> str:
    """
    Decode the datastring that is sent from the child process to the master process.

    Args:
        datastring (str): The datastring that is sent from the child process to the master process.

    Returns:
        str: The atom name of the model that finished training or False if the model did not finish training.
    """
    if datastring.endswith(ENCODING_FINISHED_SYMBOL):
        return datastring.split(ENCODING_STOP_SYMBOL)[0]
    return False

def decode_starting(datastring: str) -> str:
    """
    Decode the datastring that is sent from the child process to the master process.

    Args:
        datastring (str): The datastring that is sent from the child process to the master process.

    Returns:
        str: The atom name of the model that finished training or False if the model did not finish training.
    """
    if datastring.endswith(ENCODING_START_SYMBOL):
        return datastring.split(ENCODING_STOP_SYMBOL)[0]
    return False

def encode_data(atom_name, epoch, loss):
    return f"{atom_name}{ENCODING_STOP_SYMBOL}{epoch}{ENCODING_STOP_SYMBOL}{loss}".encode()

def encode_finished(atom_name):
    return f"{atom_name}{ENCODING_STOP_SYMBOL}{ENCODING_FINISHED_SYMBOL}".encode()

def encode_starting(atom_name):
    return f"{atom_name}{ENCODING_STOP_SYMBOL}{ENCODING_START_SYMBOL}".encode()

def ts():
    return datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")

if __name__ == "__main__":
    # Read sys args
    if len(sys.argv) != 2:
        raise Exception(f"Please provide the number of models as an argument")
    
    # To keep track of all models that are finished
    number_of_models = int(sys.argv[1])
    finished_models = []
    
    # Send initial email
    sendStartMail()
    
    # Create a TCP/IP socket
    master_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the port and localhost
    server_address = ('', PORT)

    print(ts() + ': Starting up master resolver on {}:{}'.format(*server_address))
    master_socket.bind(server_address)

    # Listen for incoming connections
    master_socket.listen(number_of_models)
    
    some_models_train = True
    while some_models_train:
        conn, addr = master_socket.accept()
        # Read data from the socket
        from_client = ''
        send_update = False
        while True:
            data = conn.recv(4096)
            if not data: break
            from_client += data.decode('utf8')
        conn.close()

        # Check if a model finished training
        if decode_finished(from_client):
            finished_models.append(decode_finished(from_client))
            print(f"{ts()}: {decode_finished(from_client)} finished training, {len(finished_models)}/{number_of_models} done")
            send_update = True
        # Check if a model started training
        elif decode_starting(from_client):
            print(f"{ts()}: {decode_starting(from_client)} starting training")
        # Check if a model is still training
        else:
            atom_name, epoch, loss = decode_data(from_client)
            print(f"{ts()}: {atom_name} epoch {epoch} with loss {loss}")
            # send_update = True
        
        # Make a subprocess to run the plot_cluster_hist.py script
        try:
            plot_cluster_hist()
        except Exception as e:
            print(f"{ts()}: Could not plot training history: {e}")

        if send_update:
            try:
                sendUpdateMail(finished_models, number_of_models)
            except Exception as e:
                print(f"{ts()}: Could not send update email: {e}")

        # Check if all models finished training
        if len(finished_models) == number_of_models:
            some_models_train = False
            print(f"{ts()}: All models finished training")
            master_socket.close()
            print(f"{ts()}: Master socket closed...")
            break