import sys
import os
import logging

#TODO: Look into making these lazy imports
import prompt_toolkit
from prompt_toolkit.completion import WordCompleter

from validphys.utils import yaml_safe
from reportengine.colors import t

from validphys.app import App
from validphys.loader import RemoteLoader
from validphys import comparefittemplates, compareclosuretemplates, reduced_comparefittemplates, bsm_only_comparefittemplate, bsm_only_with_2D_hist_comparefittemplates
from validphys.promptutils import confirm, KeywordsWithCache

log = logging.getLogger(__name__)

CURRENT_FIT_LABEL_DEFAULT = "Current Fit"
REFERENCE_FIT_LABEL_DEFAULT = "Reference Fit"


class Undefined:
    """Sentinel for arguments that need to be specified explicitly and are not.
    This is so we can have None for defaults where it makes sense.
    """

class CompareFitApp(App):
    def add_positional_arguments(self, parser):
        parser.add_argument(
            'current_fit',
            default=Undefined,
            nargs='?',
            help="The fit to produce the report for.",
        )
        parser.add_argument(
            'reference_fit',
            default=Undefined,
            nargs='?',
            help="The fit to compare with.")
        # Group together mandatory arguments that are not positional
        mandatory = parser.add_argument_group("mandatory", "Mandatory command line arguments")
        mandatory.add_argument(
            '--title', help="The title that will be indexed with the report.", default=Undefined)
        mandatory.add_argument('--author', help="The author of the report.", default=Undefined)
        mandatory.add_argument(
            '--keywords', nargs='+', help="keywords to index the report with.", default=Undefined)

        parser.add_argument(
            '--thcovmat_if_present',
            action='store_true',
            help="Use theory cov mat for calculating statistical estimators if available.")
        parser.add_argument(
            '--no-thcovmat_if_present',
            action='store_true',
            help="DEPRECATED: does nothing")
        parser.add_argument(
            '--current_fit_label',
            nargs='?',
            default=CURRENT_FIT_LABEL_DEFAULT,
            help="The label for the fit that the report is being produced for.",
        )
        parser.add_argument(
            '--reference_fit_label',
            nargs='?',
            default=REFERENCE_FIT_LABEL_DEFAULT,
            help="The label for the fit that is being compared to.")
        parser.add_argument(
            '-i',
            '--interactive',
            help="Ask interactively for the missing data",
            action='store_true')
        parser.add_argument(
            '-c',
            '--closure',
            help="Use the closure comparison template.",
            action='store_true')
        parser.add_argument(
            '-bsm',
            '--bsmonly',
            help="Use a reduced bsm only template for the comparison.",
            action='store_true')
        parser.add_argument(
            '-bsm2d',
            '--bsmonly_2d',
            help="Use a reduced bsm only template for the comparison, includes 2D histogram plots.",
            action='store_true')
        parser.add_argument(
            '-red',
            '--reduced',
            help="Use a reduced template for the comparison.",
            action='store_true')
        parser.add_argument(
            '--norm_threshold',
            default=None,
            help="Covariance matrix regularisation threshold.")

        parser.add_argument(
            '--use_fitcommondata',
            default=False,
            help="True if we are doing a closure test.")

        parser.set_defaults()

    def try_complete_args(self):
        args = self.args
        argnames = (
            'current_fit', 'reference_fit', 'title', 'author', 'keywords', 'norm_threshold', 'use_fitcommondata')
        optionalnames = (
            'current_fit_label', 'reference_fit_label')
        boolnames = (
            'thcovmat_if_present',)
        badargs = [argname for argname in argnames if args[argname] is Undefined]
        badbools = [bname for bname in boolnames if args[bname] is None]
        bad = badargs + badbools
        if bad and not args['interactive']:
            sys.exit(f"The following arguments are required: {bad}")
        try:
            for arg in bad:
                self.args[arg] = getattr(self, f'interactive_{arg}')()
            if args['interactive']:
                for arg in optionalnames:
                    self.args[arg] = getattr(self, f'interactive_{arg}')()
        except EOFError:
            raise KeyboardInterrupt()
        texts = '\n'.join(
            f'    {argname.replace("_", " ").capitalize()}: {args[argname]}'
            for argname in [*argnames, *optionalnames, *boolnames])
        log.info(f"Starting NNPDF fit comparison:\n{texts}")

    def interactive_current_fit(self):
        l = self.environment.loader
        completer = WordCompleter(l.available_fits)
        return prompt_toolkit.prompt("Enter current fit: ", completer=completer)

    def interactive_current_fit_label(self):
        #TODO Use the colors in prompt_toolkit 2+ instead of this
        default = CURRENT_FIT_LABEL_DEFAULT
        print(f"Enter label for current fit [default:\n{t.dim(default)}]:")
        #Do not use the default keyword because it is a pain to delete
        res = prompt_toolkit.prompt("")
        if not res:
            return default
        return res

    def interactive_reference_fit(self):
        l = self.environment.loader
        completer = WordCompleter(l.available_fits)
        return prompt_toolkit.prompt(
            "Enter reference fit: ", completer=completer)

    def interactive_reference_fit_label(self):
        #TODO Use the colors in prompt_toolkit 2+ instead of this
        default = REFERENCE_FIT_LABEL_DEFAULT
        print(f"Enter label for reference fit [default:\n{t.dim(default)}]:")
        #Do not use the default keyword because it is a pain to delete
        res = prompt_toolkit.prompt("")
        if not res:
            return default
        return res

    def interactive_title(self):
        #TODO Use the colors in prompt_toolkit 2+ instead of this
        default = (f"Comparison between {self.args['current_fit']} "
                   f"and {self.args['reference_fit']} ")
        print(f"Enter report title [default:\n{t.dim(default)}]:")
        #Do not use the default keyword because it is a pain to delete
        res = prompt_toolkit.prompt("")
        if not res:
            return default
        return res

    def interactive_author(self):
        default = ""
        try:
            import pwd
        except ImportError:
            pass
        else:
            default = pwd.getpwuid(os.getuid())[4]
        return prompt_toolkit.prompt("Enter author name: ", default=default)

    def interactive_norm_threshold(self):
        while True:
            res = prompt_toolkit.prompt("Enter norm threshold: ")
            if not res:
                return None
            try:
                return float(res)
            except ValueError:
                continue

    def interactive_keywords(self):
        if isinstance(self.environment.loader, RemoteLoader):
            completer = WordCompleter(words=KeywordsWithCache(self.environment.loader))
        else:
            completer = None
        kwinp = prompt_toolkit.prompt(
            "Enter keywords: ",
            completer=completer,
            complete_in_thread=True,
        )
        return [k.strip() for k in kwinp.split(',') if k]

    def interactive_thcovmat_if_present(self):
        """Interactively fill in the `use_thcovmat_if_present` runcard flag. Which is True by default
        """
        message = ("Do you want to use the theory covariance matrix, if available,\n"
                   "to calculate the statistical estimators? ")
        return confirm(message, default=True)

    def get_commandline_arguments(self, cmdline=None):
        args = super().get_commandline_arguments(cmdline)
        # This is needed because the environment wants to know how to resolve
        # the relative paths to find the templates. Best to have the template
        # look as much as possible as a runcard passed from the command line

        if args['closure']:
            #This doesn't print anything
            log.info(f"using closure test template.")
            args['config_yml'] = compareclosuretemplates.template_path

        elif args['bsmonly']:
            log.info(f"using bsm only template.")
            args['config_yml'] = bsm_only_comparefittemplate.template_path
        
        elif args['bsmonly_2d']:
            log.info(f"using bsm only template with 2D histogram plots.")
            args['config_yml'] = bsm_only_with_2D_hist_comparefittemplates.template_path
        
        elif args['reduced']:
            log.info(f"using reduced template.")
            args['config_yml'] = reduced_comparefittemplates.template_path

        else:
            args['config_yml'] = comparefittemplates.template_path
        
            
        return args

    def complete_mapping(self):
        args = self.args
        autosettings = {}
        autosettings['meta'] = {
            'title': args['title'],
            'author': args['author'],
            'keywords': args['keywords']
        }
        currentmap = {'id': args['current_fit'], 'label': args['current_fit_label']}
        autosettings['current'] = {
            'fit': currentmap,
            'pdf': currentmap,
            'theory': {
                'from_': 'fit'
            },
            'theoryid': {
                'from_': 'theory'
            },
            'speclabel': args['current_fit_label'],
            'simu_parameters': {
                'from_': 'fit'
            },
            'bsm_sector_data': {
                'from_': 'fit'
            }
        }
        refmap = {'id': args['reference_fit'], 'label': args['reference_fit_label']}
        autosettings['reference'] = {
            'fit': refmap,
            'pdf': refmap,
            'theory': {
                'from_': 'fit'
            },
            'theoryid': {
                'from_': 'theory'
            },
            'speclabel': args['reference_fit_label'],
            'simu_parameters': {
                'from_': 'fit'
            },
            'bsm_sector_data': {
                'from_': 'fit'
            }
        }
        autosettings['use_thcovmat_if_present'] = args['thcovmat_if_present']
        if args['norm_threshold'] != None:
            args['norm_threshold'] = float(args['norm_threshold'])
        autosettings['norm_threshold']=args['norm_threshold']
        if args['use_fitcommondata'] != None:
            args['use_fitcommondata'] = bool(args['use_fitcommondata'])
        autosettings['use_fitcommondata']=args['use_fitcommondata']
        return autosettings


    def get_config(self):
        self.try_complete_args()
        #No error handling here because this is our internal file
        with open(self.args['config_yml']) as f:
            #TODO: Ideally this would load round trip but needs
            #to be fixed in reportengine.
            c = yaml_safe.load(f)
        c.update(self.complete_mapping())
        return self.config_class(c, environment=self.environment)


def main():
    a = CompareFitApp()
    a.main()


if __name__ == '__main__':
    main()
