#!/usr/bin/env python3

# https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
# pip install pillow
# https://pillow.readthedocs.io/en/stable/reference
# https://pypi.org/project/ImageHash/
# pip install "thefuzz[speedup]"
# https://github.com/seatgeek/thefuzz
# https://github.com/AUTOMATIC1111/stable-diffusion-webui-tokenizer
# all:
# python -m pip install --user xmltodict "thefuzz[speedup]" pillow

# TODO
# add templates to db

import argparse
from datetime import datetime
from pathlib import Path
#from typing import Iterable
from glob import glob
from PIL import Image, UnidentifiedImageError
import sqlite3
from sqlite3 import Error
import xmltodict
import json
import sys, os
import logging
import hashlib
import time
import re
from thefuzz import fuzz, process
from pprint import PrettyPrinter
#import imagehash
from enum import Enum
import shutil
import re

DEFAULT_FNAME_PATTERN = '{file_ctime_iso}_{image_hash_short}_[app={meta_type_name}, seed={seed}, cfg={cfg_scale}, steps={steps}, sampler={sampler}, model={model}, mhash={model_hash_short}]'
DEFAULT_DB_FILE = str(Path.home()) + '/ai_meta.db'
DEFAULT_LOG_FILE = str(Path.home()) + '/ai_meta.log'
DEFAULT_LOGLEVEL_FILE = 'INFO'
DEFAULT_LOGLEVEL_CL = 'WARNING'

class Mode(Enum):
    UPDATEDB = 1
    MATCHDB = 2
    RENAME = 3
    TOJSON = 4
    TOCSV = 5
    TOKEYVALUE = 6

META_TYPE_KEY = 'meta_type'
class MetaType(Enum):
    INVOKEAI = 1
    A1111 = 2
    COMFYUI = 3

log = logging
args = None
conn = None
mode = Mode.RENAME

pp = PrettyPrinter(4, 120)


# raised when there's an error reading or processing meta data
class InvalidMeta(Exception):
    pass


# parse and return command line arguments
def args_init():
    parser = argparse.ArgumentParser(description='AIMetaDB - A Invoke-AI PNG file metadata processor')
    parser.add_argument('infile', type=Path, nargs='+',
                        help='One or more file names, directories, or glob patterns')
    parser.add_argument('--mode', type=str.upper, default='TOKEYVALUE',
                        choices=['UPDATEDB', 'MATCHDB', 'RENAME', 'TOJSON', 'TOCSV', 'TOKEYVALUE'],
                        help='Processing mode [RENAME: reame files by metadata, UPDATEDB: add file meta to db, MATCHDB: match file meta with db')
    parser.add_argument('--similarity-min', type=int, default=0,
                        help='Filter matchdb mode results based on similarity >= X [default: 0]')
    parser.add_argument('--sort_matches', action='store_true',
                        help='Sort results by similartiy (desc) grouped by infile (WARNING: memory heavy when processing large result sets)')
    parser.add_argument('--fname-pattern', type=str, default=DEFAULT_FNAME_PATTERN,
                        help='File renaming pattern for RENAME mode [default: %s]' % DEFAULT_FNAME_PATTERN)  # todo document available fields
    parser.add_argument('--dname-pattern', type=str,
                        help='After RENAME move file to the directory named by this pattern, subdirectory will be created if not existing (in local dir or --target-dir), e.g. [{file_cdate_iso}] for ./2022-01-30/')
    parser.add_argument('--no-act', action='store_true',
                        help='Only print what would be done without changing anything (mode = RENAME only)')
    parser.add_argument('--include_png_info', action='store_true',
                        help='Include full png_info when printing meta (mode = TOJSON|TOKEYVALUE only)')
    parser.add_argument('--recursive', action='store_true',
                        help='Process directories and ** glob patterns recursively')
    parser.add_argument('--target-dir', type=str,
                        help='After RENAME move file to the given target directory')
    parser.add_argument('--dbfile', type=str, default=DEFAULT_DB_FILE,
                        help='DB file location [default: %s]' % DEFAULT_DB_FILE)
    parser.add_argument('--logfile', type=str, default=DEFAULT_LOG_FILE,
                        help='Log file location [default: %s]' % DEFAULT_LOG_FILE)
    parser.add_argument('--loglevel-file', type=str.upper, default=DEFAULT_LOGLEVEL_FILE,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Log level for log file [default: %s], loglevel_cl will overwrite if higher' % DEFAULT_LOGLEVEL_FILE)
    parser.add_argument('--loglevel-cl', type=str.upper, default=DEFAULT_LOGLEVEL_CL,
                        choices=['NONE', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Log level for command line output [default: %s], NONE for quiet mode (results only)' % DEFAULT_LOGLEVEL_CL)
    global args, mode
    args = parser.parse_args()
    mode = Mode[args.mode]


# initialize logger
def log_init(logfile_path, level_file, level_cl):
    # TODO check if we can't just add a separate file handler and selt base consig to DEBUG
    # cl level can't be higher than core log level
    if (level_cl != 'NONE' and (logging.getLevelName(level_cl) < logging.getLevelName(level_file))):
        level_file = level_cl
    # https://docs.python.org/3/howto/logging.html
    logging.basicConfig(filename=logfile_path, #, encoding='utf-8',
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.getLevelName(level_file))
    global log
    log = logging.getLogger("app")
    if (level_cl != 'NONE'):
        # output logger info to stderr, any other output to stdout
        ch = logging.StreamHandler(sys.stderr)
        ch.setLevel(logging.getLevelName(level_cl))
        ch.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')) #| %(name)s
        log.addHandler(ch)


# create a database connection to a SQLite database
def db_connect(db_file):
    global conn
    try:
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row  # we want dict results (vs plain lists)
        #log.info("DB version: %s" % sqlite3.version) # TODO DEPRECATED
    except Error as e:
        log.error("Unable to create DB connection, exiting: %s" % e)
        sys.exit(1)
    # init db (if new)
    #finally:
    #    if db_conn:
    #        db_conn.close()


# initialize db if non-existing
def db_init(dbfile):
    log.info("Opening DB connection to: %s" % dbfile)
    db_connect(dbfile)
    sql_create_meta_table = """CREATE TABLE IF NOT EXISTS meta (
                                id integer PRIMARY KEY,
                                image_hash text NOT NULL UNIQUE,
                                meta_type int,
                                file_name text,
                                app_id text,
                                app_version text,
                                model text,
                                model_hash text,
                                type text,
                                prompt text,
                                steps integer,
                                cfg_scale float,
                                sampler text,
                                height integer,
                                width integer,
                                seed integer,
                                png_info text,
                                file_ctime date,
                                file_mtime date,
                                created_at date DEFAULT(DATETIME('now'))
                            ); """

    # keep large details in separate table only to be loaded if necessary
    #sql_create_meta_json_table = """CREATE TABLE IF NOT EXISTS blobs (
    #                                id integer PRIMAY KEY,
    #                                meta_id integer,
    #                                json blob,
    #                                png_info blob,
    #                                FOREIGN KEY(meta_id) REFERENCES meta(id)
    #                            );"""
    try:
        cur = conn.cursor()
        #cur.execute(f"PRAGMA foreign_keys = ON;")
        log.debug("Ensuring DB table [meta] exists.")

        # migrate
        #cur.execute('alter table meta add column meta_type integer')
        #cur.execute('alter table meta add column file_ctime date')
        #cur.execute('alter table meta add column file_mtime date')
        #cur.execute('update meta set meta_type = 1')
        #conn.commit()

        cur.execute(sql_create_meta_table);
        #log.info("Ensuring DB table exists: json")
        #cur.execute(sql_create_meta_json_table);
    except Error as e:
        log.error("Unable to initialize DB, exiting:\n%s" % e)
        sys.exit(1)


def init():
    args_init()
    log_init(args.logfile, args.loglevel_file, args.loglevel_cl)
    db_init(args.dbfile)


def file_hash(path):
    # https://stackoverflow.com/questions/22058048/hashing-a-file-in-python
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(65536) # arbitrary number to reduce RAM usage
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def a1111_meta_to_dict_to_json(params):
    #[p, s] = re.split(r'\n(Steps: )', params)   # FIXME comple all regex globally
    #try:
    result = {}
    is_prompt = True
    last_key = ""
    for l in re.split(r'\n', params):
        # first line is always prompt (w/o prefix)
        if (is_prompt):
            result['prompt'] = l
            is_prompt = False
            continue
        # at least 4 of the known core attributes in current line? (crude check)
        if (len(re.findall(r'(steps|sampler|size|seed|model hash|cfg scale): ', l, flags=re.IGNORECASE)) >= 4):
            # convert to dict
            result.update(dict(map(lambda e: [e[0].lower().strip().replace(' ', '_'), e[1].strip()], re.findall(r'[, ]*([^:]+): ([^,]+)?', l))))
        elif (re.match(r'^[\w -]+: ', l)): # TODO improve, multi-line template might have lines starting like this
            # add to dict
            key = re.sub(r'^([^:]+):.*', r'\1', l).lower().strip().replace(' ', '_')
            val = re.sub(r'^[^:]+: *(.*)', r'\1', l).strip()
            result[key] = val
            last_key = key
        else:
            # continue multi-line field (append)
            if (last_key == ""):
                continue # ignore continueation of first line (redundant prompt)
            result[last_key] += " " + l.strip()

    # does this look like a1111 meta? (crude check)
    re_exp_find = r'([{}|]|__)'
    re_exp_warn = r'(.{6}(?:[{}|]|__).{6}|(?:[{}|]|__).{6}|.{6}(?:[{}|]|__)|(?:[{}|]|__))'
    if (len(result) <= 0 or ('prompt' not in result) or ('steps' not in result)):
        raise InvalidMeta("Unable to process presumed A1111 meta:\n%s\n-> %s" % (params, result))
    if (len(re.findall(re_exp_find, result['prompt'])) > 0):
        log.info('Prompt seems to contain un-evaluated expressions, please check before re-using prompt: %s' % re.findall(re_exp_warn, result['prompt']))
    if (re.match(re_exp_find, result['negative_prompt'])):
        log.info('Prompt seems to contain un-evaluated expressions,, please check before re-using prompt: %s' % str(re.findall(re_exp_warn, result['negative_prompt'])))
    [result['width'], result['height']] = result['size'].split('x')
    result['app_id'] = 'AUTOMATIC1111/stable-diffusion-webui'
    result['app_version'] = None # info not provided
    result['type'] = None  # info not provided (t2i/i2i)
    result[META_TYPE_KEY] = MetaType.A1111.value
    return result
    #nSteps: 20, Sampler: Euler a, CFG scale: 8.5, Seed: 2518596816, Size: 512x768, Model hash: 7dd744682a'


# https://stackoverflow.com/a/14962509
def find_in_dict(obj, key):
    if key in obj: return obj[key]
    for k, v in obj.items():
        if isinstance(v,dict):
            item = find_in_dict(v, key)
            if item is not None:
                return item


def is_none_or_empty(s):
    return s is None or (isinstance(s, str) and s.strip() == "")

def get_ctime_iso_from_name_or_meta(path):
    file_name = os.path.basename(path)
    pattern = r"\d{4}-\d{2}-\d{2}T\d{6}\.\d{6}"
    match = re.search(pattern, file_name)
    if match:
        return match.group()
    else:
        return timestamp_to_iso(os.path.getctime(path))

def get_meta(path, png, image_hash, png_meta_as_dict=False, include_png_info=False):
    file_name = os.path.basename(path)
    try:
        meta_dict = png.info
        #meta_dict['xmp_json'] = json.dumps(xmltodict.parse(meta_dict['XML:com.adobe.xmp']), indent=4)
        #print(meta_dict['xmp_json'])
        if ('sd-metadata' in meta_dict):  # invoke-ai
            # parse sd-metadata (json) string to dict
            sd_meta = json.loads(meta_dict['sd-metadata'])
            sd_meta[META_TYPE_KEY] = MetaType.INVOKEAI.value
            meta_dict['sd-metadata'] = sd_meta  # overwrite json-string with dict
        elif ('parameters' in meta_dict):   # a1111
            sd_meta = a1111_meta_to_dict_to_json(meta_dict['parameters'])
            sd_meta[META_TYPE_KEY] = MetaType.A1111.value
        elif ('workflow' in meta_dict):   # comfyui
            sd_meta = {}
            sd_meta['comfyui_prompt'] = json.loads(meta_dict['prompt'])
            sd_meta['comfyui_workflow'] = json.loads(meta_dict['workflow'])
            sd_meta[META_TYPE_KEY] = MetaType.COMFYUI.value
        else:
            raise InvalidMeta("No known meta found in [file_path:\"%s\"]" % path)
    except KeyError as e:
        log.warning("No known meta found in [file_path: %s]" % path)
        raise InvalidMeta(e)
    png_info = meta_dict if png_meta_as_dict else json.dumps(meta_dict)
    m = sd_meta.copy()
    if sd_meta[META_TYPE_KEY] == MetaType.INVOKEAI.value:

        # TODO add all fields (update m with flattened keys)
        result = {"meta_type": sd_meta[META_TYPE_KEY],
                  "meta_type_name": MetaType.INVOKEAI.name,
                  "file_name": file_name,
                  "app_id": sd_meta['app_id'],
                  "app_version": sd_meta['app_version'],
                  "model": sd_meta['model_weights'],
                  "model_hash": sd_meta['model_hash'],
                  "type": sd_meta['image']['type'],
                  "prompt": sd_meta['image']['prompt'],
                  #"negative_prompt": re.sub(... , sd_meta['image']['prompt']).strip(),
                  "steps": sd_meta['image']['steps'],
                  "cfg_scale": sd_meta['image']['cfg_scale'],
                  "sampler": sd_meta['image']['sampler'],
                  "height": sd_meta['image']['height'],
                  "width": sd_meta['image']['width'],
                  "seed": sd_meta['image']['seed'],
                  "image_hash": image_hash,
                  "file_ctime_iso": get_ctime_iso_from_name_or_meta(path),
                  "file_mtime_iso": timestamp_to_iso(os.path.getmtime(path))}
    elif sd_meta[META_TYPE_KEY] == MetaType.A1111.value:
        m.update({"meta_type": sd_meta[META_TYPE_KEY],
                  "meta_type_name": MetaType.A1111.name,
                  "file_name": file_name,
                  "image_hash": image_hash,
                  "file_ctime": os.path.getctime(path), # TODO not primarily take from filename
                  "file_mtime": os.path.getmtime(path),
                  "file_ctime_iso": get_ctime_iso_from_name_or_meta(pth),
                  "file_mtime_iso": timestamp_to_iso(os.path.getmtime(path))})
        result = m;
    else:  # comfyui
        m.update({"meta_type": sd_meta[META_TYPE_KEY],
                  "meta_type_name": MetaType.COMFYUI.name,
                  "file_name": file_name,
                  "app_id": "comfyanonymous/ComfyUI",
                  "app_version": None,   # info not provided
                  "model": "",           # default if none found below
                  "model_hash": "",      # info not provided
                  "type": None,          # info not provided (t2i/i2i)
                  "prompt": "",          # default if none found below
                  "steps": "",           # default if none found below
                  "seed": "",           # default if none found below
                  "cfg_scale": "",       # default if none found below
                  #"clip_skip": "",       # default if none found below
                  "sampler": "",         # default if none found below
                  "height": png.height,  # take dimensions from png
                  "width": png.width,    # take dimensions from png
                  "image_hash": image_hash,
                  "file_ctime": os.path.getctime(path), # TODO not primarily take from filename
                  "file_mtime": os.path.getmtime(path),
                  "file_ctime_iso": get_ctime_iso_from_name_or_meta(path),
                  "file_mtime_iso": timestamp_to_iso(os.path.getmtime(path))})
        m.update({"file_cdate_iso": m['file_ctime_iso'].split("T")[0],
                  "file_mdate_iso": m['file_mtime_iso'].split("T")[0]})
        # try to find some stuff, due to it's design there could be more than
        # one result, let's just pick the first one we can find
        # TODO add exception handling since we assume a lot of things
        found_seed_node = False  # dedicated seed node found?
        found_prompt_node = False  # dedicated prompt node found?
        # workfolw meta values
        for node in m['comfyui_workflow']['nodes']:
            try:
                # prompt
                if (node['type'] in ['ttN text']) and not found_prompt_node:
                    if 'prompt' in node['title'].lower():
                        if 'neg' in node['title'].lower() and is_none_or_empty(m['negative_template']):
                            m['negative_template'] = node['widgets_values'][0]
                        elif is_none_or_empty(m['template']):
                            m['template'] = node['widgets_values'][0]
                ## model
                #if node['type'] in ['CheckpointLoaderSimple', 'ttN pipeLoader'] and is_none_or_empty(m['model']):
                #    m['model'] = node['widgets_values'][0]
                # seed from Seed node (superseeds any other seed)
                if node['type'] == 'Float' and 'cfg' in node['title'].lower() and is_none_or_empty(m['cfg_scale']):
                    m['cfg_scale'] = node['widgets_values'][0]
                if 'seed' in node['type'].lower() and not found_seed_node:
                    found_seed_node = True
                    m['seed'] = str(node['widgets_values'][0])
            except KeyError as e:
                log.info('Unable to process ComfyUI node meta, skipping: %s' % e)
                continue

        # prompt meta values
        for id in m['comfyui_prompt']:
            try:
                node = m['comfyui_prompt'][id]
                # prompt
                if (node['class_type'].startswith('CLIPTextEncode') or node['class_type'] in ['ttN text']) and not found_prompt_node:
                    # always prefer nodes with positive and negative prompt (CLIPTextEncodeWildcards3)
                    if 'positive' in node['inputs']:
                        found_prompt_node = True
                        m['prompt'] = node['inputs']['positive']
                        m['negative_prompt'] = node['inputs']['negative']
                # model
                if node['class_type'] in ['CheckpointLoaderSimple', 'ttN pipeLoader'] and is_none_or_empty(m['model']):
                    m['model'] = os.path.splitext(node['inputs']['ckpt_name'])[0]
                    #if is_none_or_empty(m['clip_skip']):
                    #    m['clip_skip'] = str(node['inputs']['clip_skip'])
                if node['class_type'] in ['CR Model Merge Stack'] and is_none_or_empty(m['model']):
                    separator = ''
                    for i in range(1, 4):
                        if (node['inputs']['switch_' + str(i)] == "On"):
                            checkpoint = node['inputs']['ckpt_name' + str(i)].rsplit( ".", 1 )[ 0 ][:15]
                            m['model'] += separator + checkpoint + '@' + str(round(node['inputs']['model_ratio' + str(i)], 2)) + '%' + str(round(node['inputs']['clip_ratio' + str(i)],2))
                            separator = '_+_'
                # seed from Seed node (superseeds any other seed)
                if node['class_type'] == 'ttN seed' and not found_seed_node:
                    found_seed_node = True
                    m['seed'] = str(node['widgets_values'][0])
                if node['class_type'] == 'Seed' and not found_seed_node:
                    found_seed_node = True
                    m['seed'] = str(node['inputs']['seed'])
                # seed from any regular sampler (no WAS since it takes it as input)
                if node['class_type'] in ['KSampler', 'KSamplerAdvanced'] and not found_seed_node and is_none_or_empty(m['seed']):
                    m['seed'] = str(node['inputs']['seed'])
                # sampler
                if node['class_type'] in ['KSampler Config (rgthree)']:   # always leading for now, overwrite existing values
                    m['steps'] = str(node['inputs']['steps_total'])
                    m['cfg_scale'] = str(node['inputs']['cfg'])
                    m['sampler'] = str(node['inputs']['sampler_name']) + '_' + str(node['inputs']['scheduler'])
                    if isinstance(node['inputs']['scheduler'], str):
                        m['sampler'] = node['inputs']['sampler_name'] + '_' + node['inputs']['scheduler']
                if (node['class_type'] in ['ttN pipeKSampler'] or node['class_type'].startswith('KSampler')) and is_none_or_empty(m['steps']):
                    if is_none_or_empty(m['seed']) and not found_seed_node:
                        try:
                            m['seed'] = str(node['inputs']['seed'][0])
                        except:
                            m['seed'] = str(node['inputs']['seed'])
                    if is_none_or_empty(m['steps']):
                        m['steps'] = str(node['inputs']['steps'])
                    if is_none_or_empty(m['cfg_scale']):
                        m['cfg_scale'] = str(node['inputs']['cfg'])
                    if is_none_or_empty(m['sampler']) and isinstance(node['inputs']['scheduler'], str):
                        m['sampler'] = node['inputs']['sampler_name'] + '_' + node['inputs']['scheduler']
            except KeyError as e:
                log.info('Unable to process ComfyUI node meta, skipping: %s' % e)
                continue
        result = m;

    if ('XML:com.adobe.xmp' in meta_dict):
        xmp = xmltodict.parse(meta_dict['XML:com.adobe.xmp'])
        result['rating'] = find_in_dict(xmp, 'xmp:Rating')
    if include_png_info:
        result['png_info'] = png_info
    log.debug('Meta extracted: %s' % pp.pformat(result))
    return result


def db_get_meta_file_name_by_hash(image_hash):
    sql_select = """SELECT file_name FROM meta WHERE image_hash = :image_hash;"""
    cur = conn.cursor()
    cur.execute(sql_select, {"image_hash": image_hash})
    row = cur.fetchone()
    return None if row is None else row[0]


def db_insert_meta(path, png, image_hash):
    file_name = os.path.basename(path)
    log.info("Inserting meta in DB for [image_hash: %s, path: \"%s\"" % (image_hash, str(path)))
    sql_insert_meta = """INSERT INTO meta (meta_type, file_name, app_id, app_version,
                                           model, model_hash, type, prompt,
                                           steps, cfg_scale, sampler,
                                           height, width, seed, png_info,
                                           image_hash, file_ctime, file_mtime)
                         VALUES (:meta_type, :file_name, :app_id, :app_version,
                                 :model, :model_hash, :type, :prompt,
                                 :steps, :cfg_scale, :sampler,
                                 :height, :width, :seed, :png_info,
                                 :image_hash, :file_ctime, :file_mtime);"""
    try:
        cur = conn.cursor()
        meta_values = get_meta(path, png, image_hash, include_png_info=True);
        log.debug("DB INSERT into meta: %s" % str(meta_values))
        cur.execute(sql_insert_meta, meta_values)
        conn.commit()
    except InvalidMeta as e:
        log.warning("Unable to read meta from [file_path: \"%s\"], skipping .." % path)
        log.debug(e)
        return;
    except sqlite3.IntegrityError:
        res = cur.execute("SELECT file_name FROM meta WHERE image_hash = ?", (image_hash,))
        # todo: compare file names
        row = res.fetchone()
        if (row[0] == file_name):
            log.info("Skipping existing entry: [hash_hase: %s, file_name_old: \"%s\", file_name_new: \"%s\"]" % (image_hash, row[0], file_name))
        else:
            log.info("Skipping existing entry: [hash_hase: %s, file_name: \"%s\"]" % (image_hash, file_name))
        log.debug("Failed to insert duplicate png into DB, existing record: %s" % str(dict((row))))
        conn.rollback()
    except Error as e:
        log.error("Failed to insert new meta into DB, transaction rollback: %s\n" % e)
        conn.rollback()


def db_update_meta(path, png, image_hash):
    log.info("Updating meta in DB for [image_hash: %s, path: \"%s\"" % (image_hash, str(path)))
    sql_update_meta = """UPDATE meta
                         SET meta_type = :meta_type, file_name = :file_name, app_id = :app_id, app_version = :app_version,
                             model = :model, model_hash = :model_hash, type = :type, prompt = :prompt,
                             steps = :steps, cfg_scale = :cfg_scale, sampler = :sampler,
                             height = :height, width = :width, seed = :seed, png_info = :png_info,
                             file_ctime = file_ctime, file_mtime = file_mtime
                         WHERE image_hash = :image_hash;"""
    try:
        cur = conn.cursor()
        meta_values = get_meta(path, png, image_hash, include_png_info=True);
        log.debug("DB UPDATE into meta: %s" % str(meta_values))
        cur.execute(sql_update_meta, meta_values)
        conn.commit()
    except InvalidMeta as e:
        log.warning("Unable to read meta from [file_path: \"%s\"], skipping .." % path)
        log.debug(e)
        return;
    except Error as e:
        log.error("Failed to update existing meta in DB, transaction rollback:\n" % e)
        conn.rollback()


def db_update_or_create_meta(path, png, image_hash):
    file_name_org = db_get_meta_file_name_by_hash(image_hash)
    if (file_name_org == None):  # not found?
        db_insert_meta(path, png, image_hash)
    else:  # record with same image hash found
        if (file_name_org != os.path.basename(path)):
            log.debug("Updating meta, file_name will change from [\"%s\"] to [\"%s\"]" %
            (file_name_org, os.path.basename(path)))
        db_update_meta(path, png, image_hash)


def print_column_headrs():
    # TODO support custom pattern
    print('in_file_idx | db_file_idx | file_source | similarity | steps | cfg_scale | sampler | height | width | seed | model_hash | model | meta_type | type | image_hash | file_name | file_ctime | file_mtime | app_id | app_version | prompt')


def sanitize_value(val, escape_quotes=True):
    val_str = str(val)
    # escape double-quotes " in prompt (promt will be within " on output)
    # replace all newline with spaces
    result = re.sub(r'(["\\])', r'\\\1', val_str) if escape_quotes else val_str
    result = re.sub(r'\r?\n', r' ', result).strip()
    return result


def timestamp_to_iso(ts):
    #try:
    return datetime.fromtimestamp(ts).isoformat()
    #except:
    #    log.debug("Unable to convert timestamp [ts=%s] to iso datetime." % ts)
    #    return ""

# convert meta dict to output tuple
def meta_to_output_tuple(dict):
    # TODO support custom patterns
    prompt_esc = re.sub(r'\r?\n', r' ', re.sub(r'(["\\])', r'\\\1', dict['prompt']).strip())
    return (dict['steps'], dict['cfg_scale'], dict['sampler'], dict['height'], dict['width'], dict['seed'],
            dict['model_hash'], dict['model'], dict[META_TYPE_KEY], dict['type'], dict['image_hash'],
            dict['file_ctime_iso'], dict['file_mtime_iso'], dict['file_name'],
            dict['app_id'], dict['app_version'], sanitize_value(prompt_esc))


def db_match(path, png, image_hash, idx, sort=False):
    result = []
    print_pattern = "%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | \"%s\" | \"%s\""
    try:
        file_meta = get_meta(path, png, image_hash)
    except InvalidMeta as e:
        log.warning("Unable to read meta from [file_path: \"%s\"], skipping .." % path)
        log.debug(e)
        return
    file_meta['prompt'] = sanitize_value(file_meta['prompt'])
    sql_select = """SELECT steps, cfg_scale, sampler, height, width, seed, model_hash, model, meta_type, type, image_hash, file_name, file_ctime, file_mtime, app_id, app_version, prompt FROM meta;"""
    cur = conn.cursor()
    cur.execute(sql_select)
    result_set = cur.fetchall()
    log.debug("Meta for file [\"%s\"]:\n%s" % (path, file_meta))
    # TODO allow file output (nice to have, redirect possible)
    file_printed = False
    i = 1
    for row in result_set:
        row_meta = dict(row)
        row_meta['prompt'] = sanitize_value(row_meta['prompt'])
        row_meta['file_ctime_iso'] = timestamp_to_iso(row_meta['file_ctime'])
        row_meta['file_mtime_iso'] = timestamp_to_iso(row_meta['file_mtime'])
        #print("-----> %s\n-----> %s" % (row_meta['prompt'], file_meta['prompt']))
        similarity = fuzz.token_sort_ratio(row_meta['prompt'], file_meta['prompt'])
        if (file_meta['image_hash'] == row_meta['image_hash']):
            log.debug("Skipping DB meta with same image_hash as given file")
            continue
        if (similarity >= args.similarity_min):
            if (not file_printed):  # print current file in first iteration (not at all if no matches were found)
                file_printed = True
                print(file_meta)
                t = (idx, i, 'file', 100) + meta_to_output_tuple(file_meta)
                if (sort):
                    result.append(t)
                else:
                    print(print_pattern % t)
            t = (idx, i, 'db', similarity) + meta_to_output_tuple(row_meta)
            if (sort):
                result.append(t)
            else:
                print(print_pattern % t)
        i = i+1
    cur.close()
    if (sort):
        for r in sorted(result, key=lambda x: (x[0], -x[3])):
            print(print_pattern % r)


def rename_file(file_path, png, image_hash):
    # FIXME split path an fname, currently filename must be first (path is included)
    try:
        meta = get_meta(file_path, png, image_hash)
    except InvalidMeta as e:
        log.warning("Unable to read meta from [file_path: \"%s\"], skipping .." % file_path)
        log.debug(e)
        return
    path = os.path.split(file_path)[0]
    [meta['file_name_noext'], meta['file_ext']] = os.path.splitext(meta['file_name'])

    # ensure that some major fields exist (with dummy value if necessary)
    for f in ['model', 'seed', 'sampler', 'cfg_scale', 'steps', 'model_hash', 'image_hash']:
        #if f not in meta or meta[f] == '': meta[f] = 'n-a'
        if f not in meta: meta[f] = ''
    # shortened versions
    meta['model_hash_short'] = meta['model_hash'][0:10]
    meta['image_hash_short'] = meta['image_hash'][0:10]
    meta['file_ctime_iso'] = meta['file_ctime_iso'].replace(':', '') # strip specials
    meta['file_mtime_iso'] = meta['file_mtime_iso'].replace(':', '') # strip specials

    try:
        out_file_name = args.fname_pattern.format(**meta) + meta['file_ext']
    except KeyError as e:
        log.warning("Unable to rename [file_path: \"%s\"] due to missing mesa field [%s], skipping ..." % (file_path, e))
        return
    out_file_name_sanitized = re.sub(r'[^,.;\[\]{}()&%#@+= \w-]', '_', out_file_name)
    out_path = os.path.normpath(os.path.join(path, out_file_name_sanitized))
    use_target_dir = False
    use_subdir = False
    if args.target_dir:
        use_target_dir = True
        out_path = os.path.normpath(os.path.join(args.target_dir, out_file_name_sanitized))
        if not Path(args.target_dir).exists():
            log.info("The --target-dir '%s' doesn't exist, trying to create it .." % args.target_dir)
            Path(args.target_dir).mkdir()
    if args.dname_pattern:
        use_subdir = True
        sub_dir = args.dname_pattern.format(**meta) + '/'
        out_dir = ''
        if use_target_dir:
            out_dir = os.path.normpath(os.path.join(args.target_dir, sub_dir))
        else:
            out_dir = os.path.normpath(os.path.join(sub_dir))
        out_path = os.path.normpath(os.path.join(out_dir, out_file_name_sanitized))
        print (out_dir)
        if not Path(out_dir).exists():
            if use_target_dir:
                log.info("The --target-dir '%s' + --dname-pattern '%s' directory '%s' doesn't exist, trying to create it ..", args.target_dir, args.dname_pattern, out_dir)
            else:
                log.info("The --dname-pattern '%s' directory '%s' doesn't exist, trying to create it ..", args.dname_pattern, out_dir)
            Path(out_dir).mkdir()
    if (os.path.normpath(file_path) == out_path):
        log.warning("Outfile identical to infile name [%s], skipping ..." % out_path)
    elif (Path(out_path).exists()):
        log.warning("File with same name exists [%s], skipping ..." % out_path)
        # TODO add --force-overwirte option
    elif (args.no_act):
        msg = "Would rename: [\"%s\"] -> [\"%s\"]" % (file_path, out_path)
        log.info(msg)
        print(msg)
    elif (use_target_dir or use_subdir):
        msg = "Moving: [\"%s\"] -> [\"%s\"]" % (file_path, out_path)
        log.info(msg)
        print(msg)
        shutil.move(file_path, out_path)
    else:
        msg = "Renaming: [\"%s\"] -> [\"%s\"]" % (file_path, out_path)
        log.info(msg)
        print(msg)
        os.rename(file_path, out_path)


def print_file_meta_json(path, png, image_hash, include_png_info=False):
    try:
        file_meta = get_meta(path, png, image_hash, png_meta_as_dict=True, include_png_info=include_png_info)
    except InvalidMeta as e:
        log.warning("Unable to read meta from [file_path: \"%s\"], skipping .." % path)
        log.debug(e)
        return
    print(json.dumps(file_meta, indent=4))


def print_file_meta_csv(path, png, image_hash):
    print_pattern = "%s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | \"%s\" | \"%s\""
    try:
        file_meta = get_meta(path, png, image_hash)
    except InvalidMeta as e:
        log.warning("Unable to read meta from [file_path: \"%s\"], skipping .." % path)
        log.debug(e)
        return
    file_meta['prompt'] = sanitize_value(file_meta['prompt'])
    print(print_pattern % meta_to_output_tuple(file_meta))


def print_file_meta_keyvalue(path, png, image_hash, include_png_info=False):
    try:
        file_meta = get_meta(path, png, image_hash, include_png_info=include_png_info)
        #file_meta.pop('png_meta')  # don't print png_meta
    except InvalidMeta as e:
        log.warning("Unable to read meta from [file_path: \"%s\"], skipping .." % path)
        log.debug(e)
        return
    for key, val in file_meta.items():
        print("%s: %s" % (key, sanitize_value(val)))


def process_file(file_path, idx):
    try:
        png = Image.open(str(file_path) )
        png.load() # needed to get for .png EXIF data
    except (AttributeError, IsADirectoryError) as e:  # directory or other type?
        log.warning("Not a file, skipping: %s" % file_path)
        log.debug(str(e))
        return
    except UnidentifiedImageError as e:
        log.warning("Not a valid image file, skipping: %s" % file_path)
        log.debug(str(e))
        return
    except OSError as e:
        log.warning("IO error while reading file, skipping: %s" % file_path)
        log.debug(str(e))
        return
    try:
        image_hash = file_hash(file_path) # TODO optimize: consider moving to later stage, may not be needed in all cases
    except OSError as e:
        log.warning("I/O error while calculate image hash for file [\"%s\"], skipping ...")
        log.debug(e)
        return
    if (mode == Mode.UPDATEDB):
        db_update_or_create_meta(file_path, png, image_hash)
    elif (mode == Mode.MATCHDB):
        print_column_headrs()
        db_match(file_path, png, image_hash, idx, args.sort_matches)
    elif (mode == Mode.RENAME):
        rename_file(file_path, png, image_hash)
    elif (mode == Mode.TOJSON):
        print_file_meta_json(file_path, png, image_hash, include_png_info=args.include_png_info)
    elif (mode == Mode.TOCSV):
        print_file_meta_csv(file_path, png, image_hash)
    elif (mode == Mode.TOKEYVALUE):
        print_file_meta_keyvalue(file_path, png, image_hash, include_png_info=args.include_png_info)
    else:  # should never happen
        log.error("Unknown mode: %s" % mode)
        sys.exit(1)


def process_paths():
    start_time_proc = time.time()
    log.info("Starting [mode=%s] ..." % args.mode)
    idx = 1
    for f in args.infile:
        start_time_path_arg = time.time()
        log.debug("Processing [file_arg: \"%s\"] ..." % f)
        # single file or glob expansion
        # FIXME currently can't handle "./" recursion (maybe others too)
        file_paths = [f] if (f.exists() and f.is_file()) else [Path(p) for p in glob(str(f.expanduser()), recursive=args.recursive)]
        if (len(file_paths) <= 0):
            log.warning("No file(s) found for infile pattern [\"%s\"], skipping ..." % f)
            continue
        for file_path in file_paths:
            start_time_file = time.time()
            log.info("Processing [#%s, file: \"%s\"] ..." % (idx, file_path))
            process_file(file_path, idx)
            log.debug("Finished processing file [#%s, exec_time: %ssec, file_path: \"%s\"]" %
                      (idx, round(time.time() - start_time_file, 3), file_path))
            idx = idx + 1
        log.debug("Finished processing file_arg [exec_time: %ssec, file_arg: \"%s\"]" %
                  (round(time.time() - start_time_path_arg, 3), f))
    log.info("Finished [mode=%s, exec_time: %ssec]!" %
             (mode.name, round(time.time() - start_time_proc, 3)))


if __name__ == '__main__':
    init()
    start_time = time.time()
    # TODO handle somewhere else
    if (mode == Mode.TOCSV):
        print('steps | cfg_scale | sampler | height | width | seed | model_hash | model | meta_type | type | image_hash | file_name | file_ctime | file_mtime | app_id | app_version | prompt')
    process_paths()
