import datetime
import hashlib
import hmac
import sys

import requests
import boto3
from boto3 import Session

session = Session()
sts = boto3.client('sts')


def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def get_signature_key(key, date_stamp, region_name, service_name):
    k_date = sign(('AWS4' + key).encode('utf-8'), date_stamp)
    k_region = sign(k_date, region_name)
    k_service = sign(k_region, service_name)
    k_signing = sign(k_service, 'aws4_request')
    return k_signing


def get_signature_headers(
        access_key,
        secret_key,
        session_token,
        url,
        method='POST',
        querystring='',
        body='',
        service='execute-api',
        region='us-east-1',
        content_type='application/json'):
    request_parameters = body

    url_split = url.split('//')[1].split('/')
    host = url_split[0]
    path = '/{}'.format('/'.join(url_split[1:]))

    if access_key is None or secret_key is None:
        print('No access key is available.')
        sys.exit()

    t = datetime.datetime.utcnow()
    amz_date = t.strftime('%Y%m%dT%H%M%SZ')
    date_stamp = t.strftime('%Y%m%d')

    canonical_uri = path
    canonical_querystring = querystring
    canonical_headers = 'content-type:' + content_type + '\n' + 'host:' + host + '\n' + 'x-amz-date:' + amz_date + '\n' + 'x-amz-security-token:' + session_token + '\n'

    signed_headers = 'content-type;host;x-amz-date;x-amz-security-token'

    payload_hash = hashlib.sha256(request_parameters.encode('utf-8')).hexdigest()

    canonical_request = method + '\n' + canonical_uri + '\n' + canonical_querystring + '\n' + canonical_headers + '\n' + signed_headers + '\n' + payload_hash

    algorithm = 'AWS4-HMAC-SHA256'
    credential_scope = date_stamp + '/' + region + '/' + service + '/' + 'aws4_request'
    string_to_sign = algorithm + '\n' + amz_date + '\n' + credential_scope + '\n' + hashlib.sha256(
        canonical_request.encode('utf-8')).hexdigest()

    signing_key = get_signature_key(secret_key, date_stamp, region, service)

    signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()

    authorization_header = algorithm + ' ' + 'Credential=' + access_key + '/' + credential_scope + ', ' + 'SignedHeaders=' + signed_headers + ', ' + 'Signature=' + signature

    headers = {'Content-Type': content_type,
               'X-Amz-Date': amz_date,
               'Authorization': authorization_header,
               'X-Amz-Security-Token': session_token}

    return headers


def post(url, payload):

    try:
        credentials = sts.get_session_token()['Credentials']
        access_key = credentials['AccessKeyId']
        secret_key = credentials['SecretAccessKey']
        session_token = credentials['SessionToken']
    except:
        credentials = session.get_credentials().get_frozen_credentials()
        access_key = credentials.access_key
        secret_key = credentials.secret_key
        session_token = credentials.token

    headers = get_signature_headers(
        access_key=access_key,
        secret_key=secret_key,
        session_token=session_token,
        url=url,
        body=payload
    )

    return requests.post(
        url=url,
        data=payload,
        headers=headers,
    )