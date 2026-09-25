// SeQan3 includes
#include <seqan3/argument_parser/all.hpp>
#include <seqan3/core/debug_stream.hpp>

// includes for reading traces from log file
#include <cstring>
#include "pugixml.hpp"
#include <iostream>

// writing on a text file
#include <iostream>
#include <fstream>

/**
 * @brief Read a trace log from given xes file
 * 
 * @param file_name Path to xes file
 * @return std::vector<std::vector<std::string>> List of traces with event names
 */
std::vector<std::vector<std::string>> read_trace_log(const char* file_name)
{
    // time measurement
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(file_name);
    if (!result) 
    {
        std::cout << "Something went wrong while reading " << file_name << "!" << std::endl;
        std::vector<std::vector<std::string>> empty;
        return empty;
    }

    std::vector<std::vector<std::string>> traces_vector;

    int bin_number = 0;

    // read every trace node and build list of events
    for (pugi::xml_node trace: doc.child("log").children("trace"))
    {
        std::string trace_string;
        std::vector<std::string> trace_vector;

        for (pugi::xml_node event: trace.children("event"))
        {
            for (pugi::xml_node event_string_node: event.children("string"))
            {
                if (std::strcmp(event_string_node.attribute("key").as_string(), "concept:name") == 0)
                {
                    trace_vector.push_back(event_string_node.attribute("value").as_string());
                    break;
                }
            }
        }
        traces_vector.push_back(trace_vector);

        bin_number++;
    }

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Reading " << file_name << " completed in " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000.0 << "[ms]" << std::endl;

    return traces_vector;
}

/**
 * @brief Convert a xes file to txt format
 * 
 * @param xes_file Path to xes file
 * @param txt_file Path to txt file
 */
void convert_xes_to_txt(const char* xes_file, const char* txt_file)
{
    std::ofstream myfile (txt_file);
    if (myfile.is_open())
    {
        // read all traces from xes file
        std::vector<std::vector<std::string>> xes_traces = read_trace_log(xes_file);

        // convert each trace into string presentation
        for (int i = 0; i < xes_traces.size(); i++)
        {
            std::string trace_str;
            for (const auto &piece : xes_traces[i]) trace_str += piece + " - ";
            trace_str = trace_str.substr(0, trace_str.size() - 3);

            myfile << trace_str << "\n";
        }

        myfile.close();
    }
    else std::cout << "Unable to open file";
}

int main(int argc, char* argv[])
{
    // initialize and fill argument parser
    seqan3::argument_parser argument_parser{"XES-to-TXT", argc, argv};

    std::filesystem::path xes_file{}, txt_file{};

    argument_parser.add_option(xes_file, 'i', "input", "The input XES file.",
                               seqan3::option_spec::required, seqan3::input_file_validator{{"xes"}});
    argument_parser.add_option(txt_file, 'o', "output", "The output TXT file. (Will override existing files!)",
                               seqan3::option_spec::required,
                               seqan3::output_file_validator{seqan3::output_file_open_options::open_or_create, {"txt"}});

    try
    {
        argument_parser.parse();
        convert_xes_to_txt(xes_file.string().c_str(), txt_file.string().c_str());
    }
    catch (seqan3::argument_parser_error const & ext)
    {
        seqan3::debug_stream << "[Error] " << ext.what() << "\n";
        return -1;
    }

    return 0;
}