"""
=================================== LICENSE ==================================
Copyright (c) 2021, Consortium Board ROXANNE
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

Neither the name of the ROXANNE nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY CONSORTIUM BOARD ROXANNE ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL CONSORTIUM BOARD TENCOMPETENCE BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
==============================================================================
"""
import pickle
import json
import itertools
import pandas as pd
import networkx as nx

from networkx.readwrite import json_graph

def isNaN(num):
    return num != num

def update_crimeid_and_offenderid(dataframe):
    df = dataframe

    if 'crimeID' in df.columns:
        list_crime_id = df['crimeID'].values
        list_temp = []
        for i, id in enumerate(list_crime_id):
            temp = 'CID#' + str(id)
            list_temp.append(temp)
        list_crime_id = list_temp
        # Update dataframe crimeID for analyzing
        df['crimeID'] = list_crime_id

    if 'offenderID' in df.columns:
        list_offender_id = df['offenderID'].values
        list_temp = []
        for i, id in enumerate(list_offender_id):
            temp = 'OID#' + str(id)
            list_temp.append(temp)
        list_offender_id = list_temp
        # Update dataframe offenderID for analyzing
        df['offenderID'] = list_offender_id

    return df

def generate_dict_crimeid_and_offenderid(df_offenders_per_crime):
    df = df_offenders_per_crime
    dict_cid = {}
    dict_oid = {}
    for index, row in df.iterrows():
        if row['crimeID'] not in dict_cid:
            set_current_oid = set()
            set_current_oid.add(row['offenderID'])
            dict_cid[row['crimeID']] = set_current_oid
        else:
            set_current_oid = dict_cid[row['crimeID']]
            set_current_oid.add(row['offenderID'])
            dict_cid[row['crimeID']] = set_current_oid

        if row['offenderID'] not in dict_oid:
            set_current_cid = set()
            set_current_cid.add(row['crimeID'])
            dict_oid[row['offenderID']]= set_current_cid
        else:
            set_current_cid = dict_oid[row['offenderID']]
            set_current_cid.add(row['crimeID'])
            dict_oid[row['offenderID']] = set_current_cid

    return dict_cid, dict_oid

def generate_dict_edges_from_dict_crime_id(dict_cid):
    dict_edges = dict()
    for key, value in dict_cid.items():
        list_temp_edge = list()
        for u, v in itertools.product(value, value):
            if u != v:
                e = (u, v)
                list_temp_edge.append(e)
                # print(e)
        
        for e in list_temp_edge:
            if e not in dict_edges:
                dict_edges[e] = 1.0
            else:
                weight = dict_edges[e]
                weight = weight + 1
                dict_edges[e] = weight
    return dict_edges

def generate_nx_graph(graph_name, graph_description, graph_version, graph_id, dict_cid, dict_edges, df_crimes_data):
    G = nx.Graph(name=graph_name, description=graph_description, version=graph_version, id=graph_id)

    set_cid_in_data = set()
    for index, row in df_crimes_data.iterrows():
        if row['crimeID'] in dict_cid.keys():
            node_id = row['crimeID']
            list_oid = dict_cid[row['crimeID']]
            x_coordinate = row['x']
            y_coordinate = row['y']
            date = index
            emb_case_summary = row['txt_emb_1']
            emb_stolen_items_description = row['txt_emb_2']
            emb_victim_testimony = row['txt_emb_3']
            crime_type = row['z']

            # remove nodes with emb_case_summary being NaN
            if not isNaN(emb_case_summary):
                G.add_node(node_id, type='crime',
                            list_oid=list(list_oid),
                            x_coordinate=x_coordinate,
                            y_coordinate=y_coordinate,
                            date=date.strftime('%Y-%m-%d'),
                            emb_case_summary=emb_case_summary,
                            emb_stolen_items_description=emb_stolen_items_description,
                            emb_victim_testimony=emb_victim_testimony,
                            crime_type=crime_type,
                            original_node=node_id)
            
            set_cid_in_data.add(node_id)

    for key, value in dict_edges.items():
        u, v = key
        weight = value

        if u in G.nodes() and v in G.nodes():
            if u in set_cid_in_data and v in set_cid_in_data:
                original_edge = (u, v)
                G.add_edge(u, v, weight=weight, type='relation', observed=True, original_edge=original_edge)

    return G

def dump_nx_network_to_json(filepath_graph, nx_graph):
    json_data = json_graph.node_link_data(nx_graph)

    with open(filepath_graph, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)
    
    print('Dumped successfuly:', filepath_graph)


if __name__ == '__main__':
    df_offenders_per_crime = pd.read_csv('dataset/israel_lea_inp_burglary/raw/roxan_burglary_dataset_v2_offenders_per_crime.csv', delimiter=';')
    # print(df_offenders_per_crime.columns.values)
    df_offenders_per_crime = update_crimeid_and_offenderid(df_offenders_per_crime)
    print(df_offenders_per_crime)

    df_crimes_data = pd.read_csv('dataset/israel_lea_inp_burglary/raw/roxan_burglary_dataset_v2_crimes_data.csv', delimiter=';')
    df_crimes_data = update_crimeid_and_offenderid(df_crimes_data)

    # print(df_crimes_data)

    start_date = None
    end_date = None

    pd.set_option('display.max_columns', None)

    df_crimes_data['date'] = pd.to_datetime(df_crimes_data['date'])
    df_crimes_data = df_crimes_data.set_index(['date'])

    # start_date = '2010-01-01'
    # end_date = '2018-12-31'
    # start_date = '2020-01-01'
    # end_date = '2020-06-30'

    if start_date is not None and end_date is not None:
        df_crimes_data = df_crimes_data.loc[start_date:end_date]

    print(df_crimes_data)

    # result = df_offenders_per_crime.head(1000)
    # print(result)

    dict_cid, dict_oid = generate_dict_crimeid_and_offenderid(df_offenders_per_crime)

    # print('dict_cid:', dict_cid)
    # print('dict_oid:', dict_oid)

    dict_edges = generate_dict_edges_from_dict_crime_id(dict_oid)
    # print('dict_edges:', dict_edges)

    if start_date is not None and end_date is not None:
        graph_name = 'Israel Burglary v2 - Crime Network' + ' from ' + start_date + ' to ' + end_date
        graph_description = 'Crime Network built-on Israel Burglary dataset v2' + ' from ' + start_date + ' to ' + end_date
        graph_version = 1.0
        graph_id = 'israel-burglary-crime' + '_from_' + start_date + '_to_' + end_date
        filepath_graph = 'dataset/israel_lea_inp_burglary/israel_lea_inp_burglary_v2_crime_id_network' + \
                         '_from_' + start_date + '_to_' + end_date + '.json'
    else:
        graph_name = 'Israel Burglary v2 - Crime Network'
        graph_description = 'Crime Network built-on Israel Burglary dataset v2'
        graph_version = 1.0
        graph_id = 'israel-burglary-crime'
        filepath_graph = 'dataset/israel_lea_inp_burglary/israel_lea_inp_burglary_v2_crime_id_network.json'

    nx_g = generate_nx_graph(graph_name, graph_description, graph_version, graph_id, dict_cid, dict_edges, df_crimes_data)
    print(nx.info(nx_g))
    # print(nx_g.nodes(data=True))
    dump_nx_network_to_json(filepath_graph, nx_g)
