// SeQan3 includes
#include <seqan3/argument_parser/all.hpp>
#include <seqan3/core/debug_stream.hpp>

// includes for reading traces from log file
#include "pugixml.hpp"
#include <iostream>

// reading from a text file
#include <iostream>
#include <fstream>

/**
 * @brief Convert a txt file to xes format
 * 
 * @param txt_file Path to txt file
 * @param xes_file Path to xes file
 */
void convert_txt_to_xes(const char* txt_file, const char* xes_file)
{
    std::ifstream file(txt_file);
    std::string str;

    if (file.is_open())
    {
        // create template XES
        pugi::xml_document doc;

        auto declarationNode = doc.append_child(pugi::node_declaration);
        declarationNode.append_attribute("version") = "1.0";
        declarationNode.append_attribute("encoding") = "UTF-8";

        auto root = doc.append_child("log");
        root.append_attribute("xes.version") = "1.0";
        root.append_attribute("xes.features") = "nested-attributes";
        root.append_attribute("openxes.version") = "1.0RC7";

        auto extension = root.append_child("extension");
        extension.append_attribute("name") = "Organizational";
        extension.append_attribute("prefix") = "org";
        extension.append_attribute("uri") = "http://www.xes-standard.org/org.xesext";

        extension = root.append_child("extension");
        extension.append_attribute("name") = "Time";
        extension.append_attribute("prefix") = "time";
        extension.append_attribute("uri") = "http://www.xes-standard.org/time.xesext";

        extension = root.append_child("extension");
        extension.append_attribute("name") = "Lifecycle";
        extension.append_attribute("prefix") = "lifecycle";
        extension.append_attribute("uri") = "http://www.xes-standard.org/lifecycle.xesext";

        extension = root.append_child("extension");
        extension.append_attribute("name") = "Semantic";
        extension.append_attribute("prefix") = "semantic";
        extension.append_attribute("uri") = "http://www.xes-standard.org/semantic.xesext";

        extension = root.append_child("extension");
        extension.append_attribute("name") = "Concept";
        extension.append_attribute("prefix") = "concept";
        extension.append_attribute("uri") = "http://www.xes-standard.org/concept.xesext";

        auto global = root.append_child("global");
        global.append_attribute("scope") = "trace";
        auto global_string = global.append_child("string");
        global_string.append_attribute("key") = "concept:name";
        global_string.append_attribute("value") = "__INVALID__";

        global = root.append_child("global");
        global_string = global.append_child("string");
        global_string.append_attribute("key") = "concept:name";
        global_string.append_attribute("value") = "__INVALID__";
        global_string = global.append_child("string");
        global_string.append_attribute("key") = "lifecycle:transition";
        global_string.append_attribute("value") = "complete";

        auto classifier = root.append_child("classifier");
        classifier.append_attribute("name") = "MXML Legacy Classifier";
        classifier.append_attribute("keys") = "concept:name lifecycle:transition";

        classifier = root.append_child("classifier");
        classifier.append_attribute("name") = "Event Name";
        classifier.append_attribute("keys") = "concept:name";

        classifier = root.append_child("classifier");
        classifier.append_attribute("name") = "Resource";
        classifier.append_attribute("keys") = "org:resource";

        classifier = root.append_child("classifier");
        classifier.append_attribute("name") = "Lifecycle transition";
        classifier.append_attribute("keys") = "lifecycle:transition";

        auto root_string = root.append_child("string");
        root_string.append_attribute("key") = "concept:name";
        root_string.append_attribute("value") = "Event Log";

        root_string = root.append_child("string");
        root_string.append_attribute("key") = "lifecycle:model";
        root_string.append_attribute("value") = "standard";


        int i = 0;
        while (std::getline(file, str))
        {
            // add into xes file
            auto trace_node = root.append_child("trace");

            auto string_node = trace_node.append_child("string");
            string_node.append_attribute("key") = "concept:name";
            string_node.append_attribute("value") = ("trace_" + std::to_string(i)).c_str();

            // split string into single events
            std::string delimiter = " - ";
            size_t pos = 0;
            std::string token;
            while ((pos = str.find(delimiter)) != std::string::npos)
            {
                token = str.substr(0, pos);
                
                /*
                Example trace
                <trace>
                    <string key="concept:name" value="trace_0"/>
                    <event>
                        <string key="org:resource" value="NONE"/>
                        <string key="concept:name" value="Create Fine"/>
                        <string key="lifecycle:transition" value="complete"/>
                        <date key="time:timestamp" value="2022-07-26T23:52:35.144+02:00"/>
                    </event>
                    <event>
                        <string key="org:resource" value="NONE"/>
                        <string key="concept:name" value="Appeal to Judge"/>
                        <string key="lifecycle:transition" value="complete"/>
                        <date key="time:timestamp" value="2022-07-27T00:52:35.144+02:00"/>
                    </event>
                </trace>
                */
                auto event_node = trace_node.append_child("event");
                auto string_node1 = event_node.append_child("string");
                string_node1.append_attribute("key") = "org:resource";
                string_node1.append_attribute("value") = "NONE";
                auto string_node2 = event_node.append_child("string");
                string_node2.append_attribute("key") = "concept:name";
                string_node2.append_attribute("value") = token.c_str();
                auto string_node3 = event_node.append_child("string");
                string_node3.append_attribute("key") = "lifecycle:transition";
                string_node3.append_attribute("value") = "complete";

                auto date_node = event_node.append_child("date");
                date_node.append_attribute("key") = "time:timestamp";
                date_node.append_attribute("value") = "2022-07-27T00:52:35.144+02:00";

                str.erase(0, pos + delimiter.length());
            }
            // add last event
            auto event_node = trace_node.append_child("event");
            auto string_node1 = event_node.append_child("string");
            string_node1.append_attribute("key") = "org:resource";
            string_node1.append_attribute("value") = "NONE";
            auto string_node2 = event_node.append_child("string");
            string_node2.append_attribute("key") = "concept:name";
            string_node2.append_attribute("value") = str.c_str();
            auto string_node3 = event_node.append_child("string");
            string_node3.append_attribute("key") = "lifecycle:transition";
            string_node3.append_attribute("value") = "complete";

            i++;
        }

        // save to file
        bool saveSucceeded = doc.save_file(xes_file, PUGIXML_TEXT("  "));
    }
}

int main(int argc, char* argv[])
{
    // initialize and fill argument parser
    seqan3::argument_parser argument_parser{"TXT-to-XES", argc, argv};

    std::filesystem::path txt_file{}, xes_file{};

    argument_parser.add_option(txt_file, 'i', "input", "The input TXT file.",
                               seqan3::option_spec::required, seqan3::input_file_validator{{"txt"}});
    argument_parser.add_option(xes_file, 'o', "output", "The output XES file. (Will override existing files!)",
                               seqan3::option_spec::required,
                               seqan3::output_file_validator{seqan3::output_file_open_options::open_or_create, {"xes"}});

    try
    {
        argument_parser.parse();
        convert_txt_to_xes(txt_file.string().c_str(), xes_file.string().c_str());
    }
    catch (seqan3::argument_parser_error const & ext)
    {
        seqan3::debug_stream << "[Error] " << ext.what() << "\n";
        return -1;
    }

    return 0;
}