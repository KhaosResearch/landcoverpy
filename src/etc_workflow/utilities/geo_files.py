def _kmz_to_geojson(kmz_file: str) -> str:
    """
    Transform a kmz file to a geojson file
    """
    import fiona
    fiona.drvsupport.supported_drivers['KML'] = 'rw'
    geojson_file = kmz_file[:-4] + ".geojson"
    with ZipFile(kmz_file, "r") as zip_in:
        zip_in.extractall("./databases/")
    df = gpd.read_file(filename="./databases/doc.kml", driver="KML")
    df.to_file(geojson_file, driver="GeoJSON")

    _postprocess_geojson_file(geojson_file)

    return geojson_file

def _postprocess_geojson_file(geojson_file: str):
    """
    Postprocess a geojson that comes from a kmz, transform its html table to a dictionary
    """
    geojson = read_geojson(geojson_file)

    for feature in geojson['features']:

        propery = feature["properties"]

        html_table = propery["Description"]

        if 'html' in html_table:
            html_table_splitted = html_table.split('</td> </tr> <tr> <td> ')
            html_table_without_header = html_table_splitted[1]
            html_table_without_header_splitted = html_table_without_header.split('</td> </tr> </table> </body>')
            content = html_table_without_header_splitted[0]
            
            parser_object = BeautifulSoup(content, 'lxml') 
            key_value_list = parser_object.find_all('td') 
            key_value_list_text=[element.get_text() for element in key_value_list]

            key_value_list_text = np.array(key_value_list_text)
            pairs= list(range(0,len(key_value_list_text),2))
            evens= list(range(1,len(key_value_list_text),2))
            keys=key_value_list_text[pairs]
            values=key_value_list_text[evens]

        else:
            keys=""
            values=""

        parsed_description={}
        for c in range(len(values)):
            parsed_description[keys[c]]= values[c]
            

        del propery["Description"]
        propery.update(parsed_description)

    with open(geojson_file, "w", encoding='utf8') as f:
        json.dump(geojson, f, ensure_ascii=False)

        