def ktools_patch(InstallKtoolsMixin: type):
    
    def install_ktools(self):
        bin_install_kwargs = self.try_get_bin_install_kwargs()
        self.install_ktools_source()

    InstallKtoolsMixin.install_ktools = install_ktools

def build_ktools(self, extract_location, system_os):
        self.announce('Building ktools', INFO)
        print('Installing ktools from source')
        print(f' :::::  system_os {system_os} :::::')

        build_dir = os.path.join(extract_location, 'ktools-{}'.format(KTOOLS_VERSION))

        system_os_flag = '--enable-osx ' if system_os == 'Darwin' else ''
        use_long = os.environ.get('AREAPERIL_TYPE', 'i8')
        use_double = os.environ.get('OASIS_FLOAT', 'u4')
        
        define = ""
        if use_double or use_long:
            if use_double:
                print(f' :::::  using double precision {use_double} :::::')
            if use_long:
                print(f' :::::  using long precision {use_long} :::::')
            
            define = "CPPFLAGS=\"{} {}\"".format("-D OASIS_FLOAT_TYPE_DOUBLE" if use_double else "", "-D AREAPERIL_TYPE_UNSIGNED_LONG_LONG" if use_long else "")

        exit_code = os.system(f'cd {build_dir} && ./autogen.sh && ./configure {system_os_flag} && make {define}')
        if (exit_code != 0):
            print('Ktools build failed.\n')
            sys.exit(1)
        return build_dir

    InstallKtoolsMixin.build_ktools = build_ktools

    return InstallKtoolsMixin