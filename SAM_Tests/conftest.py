import pytest
import pickle
import getpass
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
send_email_bool = False


def send_email(passw, fname):
    fromaddr = "daniel.james.camilleri.13@gmail.com"
    toaddr = "daniel.james.camilleri.13@gmail.com"

    msg = MIMEMultipart()

    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Testing Results"

    body = "Testing results for SAM"

    msg.attach(MIMEText(body, 'plain'))

    filename = fname
    attachment = open("./" + fname, "rb")

    part = MIMEBase('application', 'octet-stream')
    part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    msg.attach(part)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(fromaddr, passw)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()


@pytest.fixture(scope="module")
def get_error_list(request):
    print "setting up stuff"
    error_list = []
    if send_email_bool:
        passw = getpass.getpass(prompt="Email password")

    def teardown():
        pickled_info = {"error_list": error_list}
        pickle.dump(pickled_info, open("testing_save.pickle", "wb"))
        if send_email_bool:
            send_email(passw, "testing_save.pickle")

    request.addfinalizer(teardown)
    return error_list


def pytest_generate_tests(metafunc):
    idlist = []
    argvalues = []
    argnames = metafunc.cls.scenario_keys
    for idx, scenario in enumerate(metafunc.cls.scenario_parameters):
        idlist.append(str(idx))
        argvalues.append([scenario])
    metafunc.parametrize(argnames, argvalues, ids=idlist, scope="class")

