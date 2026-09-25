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
 * @brief Remove duplicate traces from text file
 * 
 * @param txt_file Path to txt file
 * @param deduplicate_txt_file Path to deduplicated txt file to write to
 */
void remove_double_traces(const char* txt_file, const char* deduplicate_txt_file)
{
    // initialize variables
    std::ifstream file(txt_file);
    std::ofstream output_file(deduplicate_txt_file);
    std::string str;

    if (file.is_open() && output_file.is_open())
    {
        std::vector<std::string> already_known_traces;

        while (std::getline(file, str))
        {
            // only add trace to output file if not already found
            if (std::find(already_known_traces.begin(), already_known_traces.end(), str) == already_known_traces.end())
            {
                already_known_traces.push_back(str);
                output_file << str << "\n";
            }
        }

        file.close();
        output_file.close();
    }
}

int main(int argc, char* argv[])
{
    // initialize and fill argument parser
    seqan3::argument_parser argument_parser{"Remove-double-traces", argc, argv};

    std::filesystem::path txt_file{}, deduplicate_txt_file{};

    argument_parser.add_option(txt_file, 'i', "input", "The input TXT file.",
                               seqan3::option_spec::required, seqan3::input_file_validator{{"txt"}});
    argument_parser.add_option(deduplicate_txt_file, 'o', "output", "The output XES file. (Will override existing files!)",
                               seqan3::option_spec::required,
                               seqan3::output_file_validator{seqan3::output_file_open_options::open_or_create, {"xes"}});

    try
    {
        argument_parser.parse();
        remove_double_traces(txt_file.string().c_str(), deduplicate_txt_file.string().c_str());
    }
    catch (seqan3::argument_parser_error const & ext)
    {
        seqan3::debug_stream << "[Error] " << ext.what() << "\n";
        return -1;
    }

    return 0;
}